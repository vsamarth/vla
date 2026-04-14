import json
import os
import torch
from PIL import Image
import torchvision.transforms as T
from laq.laq_model import LatentActionQuantization

def load_metadata():
    """
    Load metadata for Something-Something v2 (SthV2) dataset.
    
    Returns:
        vid_to_label: Dictionary mapping video ID to the template string (without brackets).
        label_to_id: Dictionary mapping template string to its integer index.
    """
    labels_file = "labels.json"
    train_file = "train.json"
    
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"{labels_file} not found.")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"{train_file} not found.")
    
    with open(labels_file, "r") as f:
        labels_raw = json.load(f)
    
    # Map label string (template without brackets) to its integer index
    label_to_id = {k: int(v) for k, v in labels_raw.items()}
    
    vid_to_label = {}
    with open(train_file, "r") as f:
        train_data = json.load(f)
        for item in train_data:
            # Replace [something] with something
            template = item["template"].replace("[", "").replace("]", "")
            vid_to_label[item["id"]] = template
            
    return vid_to_label, label_to_id

def get_laq_model(checkpoint_path, device):
    """
    Initialize LAQ model and load checkpoint.
    """
    laq = LatentActionQuantization(
        dim=1024,
        quant_dim=32,
        codebook_size=8,
        image_size=256,
        patch_size=32,
        spatial_depth=8,
        temporal_depth=8,
        dim_head=64,
        heads=16,
        code_seq_len=4,
        device=device,
    )
    laq.load(checkpoint_path)
    laq.eval()
    return laq

def get_latent_action(laq, frame1_path, frame2_path, transform, device):
    """
    Compute latent action between two frames.
    """
    img1 = Image.open(frame1_path).convert('RGB')
    img2 = Image.open(frame2_path).convert('RGB')
    
    t1 = transform(img1)
    t2 = transform(img2)
    
    # Concatenate frames into a single tensor of shape (1, 3, 2, 256, 256)
    # (batch, channels, frames, height, width)
    input_tensor = torch.stack([t1, t2], dim=1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        indices = laq.inference(input_tensor, return_only_codebook_ids=True)
        
    return indices[0].cpu().tolist()

if __name__ == "__main__":
    vid_to_label, label_to_id = load_metadata()
    print(f"Loaded {len(vid_to_label)} video mappings and {len(label_to_id)} action categories.")
