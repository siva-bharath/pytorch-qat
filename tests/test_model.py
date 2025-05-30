import torch
import pytest
from QatWbHpo import MobileNetV2

def test_model_initialization():
    model = MobileNetV2(num_classes=102)
    assert isinstance(model, MobileNetV2)
    assert model.classifier[1].out_features == 102

def test_model_forward():
    model = MobileNetV2(num_classes=102)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert output.shape == (2, 102)

def test_model_fusion():
    model = MobileNetV2(num_classes=102)
    model.fuse_model()
    # Add assertions to verify fusion if needed

def test_quantization_stubs():
    model = MobileNetV2(num_classes=102)
    assert hasattr(model, 'quant')
    assert hasattr(model, 'dequant')
    assert isinstance(model.quant, torch.quantization.QuantStub)
    assert isinstance(model.dequant, torch.quantization.DeQuantStub) 