import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Define a simple model with QAT support
class SimpleQATModel(nn.Module):
    def __init__(self):
        super(SimpleQATModel, self).__init__()
        # Add quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # Define the model layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1600, 10)

    def forward(self, x):
        # Quantize input
        x = self.quant(x)
        
        # Forward pass
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        
        # Dequantize output
        x = self.dequant(x)
        return x

def prepare_model_for_qat(model):
    # Specify quantization configuration
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # Prepare the model for QAT
    model_prepared = torch.quantization.prepare_qat(model)
    return model_prepared

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = SimpleQATModel()
    model = model.to(device)
    
    # Prepare model for QAT
    model = prepare_model_for_qat(model)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Load and prepare data (example with MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Train the model
    print("Starting QAT training...")
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Convert to quantized model
    model.eval()
    model_quantized = torch.quantization.convert(model)
    print("Model converted to quantized format")

if __name__ == "__main__":
    main()



