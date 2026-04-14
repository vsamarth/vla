import torch
import pytest
from generate_laq_dataset import get_laq_model, get_latent_action
import os
from PIL import Image
import torchvision.transforms as T

def test_get_laq_model():
    checkpoint_path = "laq_checkpoints/laq_openx.pt"
    if not os.path.exists(checkpoint_path):
        pytest.skip("Checkpoint not found")
    
    device = "cpu"
    laq = get_laq_model(checkpoint_path, device)
    assert laq is not None
    assert isinstance(laq, torch.nn.Module)
    assert not laq.training

def test_get_latent_action():
    checkpoint_path = "laq_checkpoints/laq_openx.pt"
    if not os.path.exists(checkpoint_path):
        pytest.skip("Checkpoint not found")
    
    device = "cpu"
    laq = get_laq_model(checkpoint_path, device)
    
    # Create dummy images
    img1_path = "test_img1.jpg"
    img2_path = "test_img2.jpg"
    Image.new('RGB', (256, 256), color='red').save(img1_path)
    Image.new('RGB', (256, 256), color='blue').save(img2_path)
    
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    
    try:
        indices = get_latent_action(laq, img1_path, img2_path, transform, device)
        assert isinstance(indices, list)
        assert len(indices) == 4
        assert all(isinstance(i, int) for i in indices)
    finally:
        if os.path.exists(img1_path):
            os.remove(img1_path)
        if os.path.exists(img2_path):
            os.remove(img2_path)
