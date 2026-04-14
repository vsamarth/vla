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
    
    # Test model loading and inference
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    checkpoint_path = "laq_checkpoints/laq_openx.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading LAQ model from {checkpoint_path}...")
        laq = get_laq_model(checkpoint_path, device)
        print("Model loaded successfully.")
        
        # Test inference on a sample pair of frames
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        
        sample_video_id = "102148"
        sample_dir = f"data/sthv2_subset/{sample_video_id}"
        frame1 = os.path.join(sample_dir, "img_00000.jpg")
        frame2 = os.path.join(sample_dir, "img_00005.jpg")
        
        if os.path.exists(frame1) and os.path.exists(frame2):
            print(f"Testing inference on {frame1} and {frame2}...")
            indices = get_latent_action(laq, frame1, frame2, transform, device)
            print(f"Latent action indices: {indices}")
            assert len(indices) == 4, f"Expected 4 indices, got {len(indices)}"
            print("Inference test passed.")
        else:
            print(f"Sample frames not found in {sample_dir}, skipping inference test.")
    else:
        print(f"Checkpoint not found at {checkpoint_path}, skipping model test.")
