# Post-training static quantization (PTQ)
import os
import sys
import time 
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

from torch.quantization import QuantStub, DeQuantStub

import warnings
from tqdm import tqdm
import gc

import optuna
from optuna.trial import TrialState

warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    module=r".*"
)
warnings.filterwarnings(
    action="default",
    module= r'torch.ao.quantization'
)

torch.manual_seed(191009)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride
                      ,padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.ReLU(inplace=False)  # Changed back to ReLU for quantization
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        # for the sake of simplicity in the below code 
        # it doesnt account when there is different input and output channels
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        # t = expand_ratio, c = number of output channels, n = layer repitition, s = stride
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.mean([2, 3]) # Global average pooling
        x = self.classifier(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self, is_qat=False):
        """Fuse Conv+BN and Conv+BN+Relu modules prior to quantization."""
        # First fuse Conv+BN pairs
        for m in self.modules():
            if type(m) == ConvBNReLU:
                # Fuse Conv+BN first
                torch.ao.quantization.fuse_modules(m, ['0', '1'], inplace=True)
                # Then fuse with ReLU6
                torch.ao.quantization.fuse_modules(m, ['0', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        # Fuse Conv+BN pairs
                        torch.ao.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches, device):
    """Evaluate model with progress tracking."""
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Evaluating')
        for image, target in pbar:
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            pbar.set_postfix({'Acc@1': f'{top1.avg:.2f}%', 'Acc@5': f'{top5.avg:.2f}%'})
            if cnt >= neval_batches:
                return top1, top5
    return top1, top5

def load_model(model_file, num_classes=102):
    """Load and modify pre-trained model with error handling."""
    try:
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
            
        # Load the pre-trained model with weights_only=True for security
        model = MobileNetV2(num_classes=1000)  # Load with 1000 classes first
        state_dict = torch.load(model_file, weights_only=True)
        model.load_state_dict(state_dict)
        
        # Just modify the output size of the final layer
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        nn.init.normal_(model.classifier[1].weight, 0, 0.01)
        nn.init.zeros_(model.classifier[1].bias)
        
        model.to('cpu')
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def prepare_data_loaders(data_path, train_batch_size, eval_batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Added data augmentation
        transforms.ToTensor(),
        normalize,
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load Oxford Flowers dataset
    dataset = torchvision.datasets.Flowers102(
        root=data_path,
        split='train',
        download=True,
        transform=train_transform
    )
    
    dataset_test = torchvision.datasets.Flowers102(
        root=data_path,
        split='test',
        download=True,
        transform=test_transform
    )
    
    # Create a larger subset for training (80% of the data)
    train_size = int(0.8 * len(dataset))
    indices = torch.randperm(len(dataset))[:train_size]
    dataset = torch.utils.data.Subset(dataset, indices)
    
    # Create a larger subset for testing (80% of the data)
    test_size = int(0.8 * len(dataset_test))
    indices = torch.randperm(len(dataset_test))[:test_size]
    dataset_test = torch.utils.data.Subset(dataset_test, indices)
    
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=train_batch_size,
        sampler=train_sampler
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=eval_batch_size,
        sampler=test_sampler
    )
    
    return data_loader, data_loader_test


data_path = 'data/flowers102'
saved_model_dir = 'data/'
float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

train_batch_size = 32
eval_batch_size = 32

data_loader, data_loader_test = prepare_data_loaders(data_path, train_batch_size, eval_batch_size)
criterion = nn.CrossEntropyLoss()

# Clear any existing CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Load the pre-trained model and modify it for Flowers102
try:
    float_model = load_model(saved_model_dir + float_model_file, num_classes=102)
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# float_model.to(device)

# # Freeze backbone, train only classifier
# for param in float_model.parameters():
#     param.requires_grad = False
# for param in float_model.classifier.parameters():
#     param.requires_grad = True

# # Set model to eval mode before fusion
# float_model.eval()

# Fuse modules before training
# print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
# float_model.fuse_model()
# print('\n Inverted Residual Block: After fusion\n\n', float_model.features[1].conv)

# Set back to train mode for fine-tuning
# float_model.train()

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, best_acc=0.0):
    """Fine-tunes the classifier layer for one epoch with progress tracking."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
               
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(idx+1):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return 100.*correct/total


def objective(trial):
    """
    Single Optuna trial that:
      • Builds a fresh MobileNetV2 classifier head
      • Samples optimiser / LR / weight-decay / dropout
      • Runs a short fine-tuning and returns val accuracy
    """
    # -----  hyper-parameters to search  ---------------------------------
    lr           = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    opt_name     = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])
    momentum     = trial.suggest_float("momentum", 0.7, 0.99) if opt_name == "SGD" else None
    dropout_p    = trial.suggest_float("dropout", 0.0, 0.5)
    epochs       = trial.suggest_int("finetune_epochs", 5, 50)
    # --------------------------------------------------------------------

    # ── fresh copy of the base model ─────────────────────
    model = load_model(saved_model_dir + float_model_file, num_classes=102)
    model.to(device)
    
    # Set to eval mode before fusion
    model.eval()
    model.fuse_model()
    
    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    #  Replace dropout with the sampled p
    model.classifier[0] = nn.Dropout(dropout_p)

    #  Pick optimiser
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(
            model.classifier.parameters(), lr=lr,
            momentum=momentum, weight_decay=weight_decay
        )
    else:  # AdamW
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), lr=lr,
            weight_decay=weight_decay
        )

    # Set back to train mode for training
    model.train()

    #  Short training loop (smaller epoch count = faster search)
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, criterion, optimizer,
                        data_loader, device, epoch)
        # --- quick val pass ------------------------------------------------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, targets in data_loader_test:
                imgs, targets = imgs.to(device), targets.to(device)
                logits = model(imgs)
                correct += logits.argmax(1).eq(targets).sum().item()
                total   += targets.size(0)
        val_acc = 100. * correct / total
        best_val_acc = max(best_val_acc, val_acc)

        # report & prune
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # clear GPU RAM between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return best_val_acc

print("\n[ Optuna search started ]")
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    sampler=optuna.samplers.TPESampler(multivariate=True, group=True)
)
study.optimize(objective, n_trials=30, timeout=None, show_progress_bar=True)

print("Search finished. Best trial:")
trial = study.best_trial
for k, v in trial.params.items():
    print(f"  {k}: {v}")
print(f"  best Val Acc: {trial.value:.2f}%")


### Evaluation
num_eval_batches = 40
print("\nSize of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches, device=device)
print('\nEvaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))

# Save final model
float_model.cpu()
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

# Final cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
