import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from QatWbHpo import MobileNetV2

def calculate_flops(model):
    model.eval()
    input_size= torch.randn(1, 3, 224, 224)
    flop_count = FlopCountAnalysis(model, input_size)
    params = parameter_count(model)
    print(f"Total FLOPs: {flop_count.total(): ,}")
    print(f"Total Parameters: {params['']:,}")

if __name__ == "__main__":
    model = MobileNetV2(num_classes=102)
    calculate_flops(model)
