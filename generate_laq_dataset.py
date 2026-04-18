import json
import os
import torch
from PIL import Image
import torchvision.transforms as T
from laq.laq_model import LatentActionQuantization
from tqdm import tqdm

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

def main():
    stride = 5
    checkpoint_path = "laq_checkpoints/laq_openx.pt"
    data_dir = "data/sthv2_subset"
    output_file = "dataset.jsonl"
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load metadata
    print("Loading metadata...")
    vid_to_label, label_to_id = load_metadata()
    
    # Initialize LAQ model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    print(f"Loading LAQ model from {checkpoint_path}...")
    laq = get_laq_model(checkpoint_path, device)
    
    # Initialize transform
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    
    video_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"Found {len(video_folders)} videos in {data_dir}")
    
    with open(output_file, "w") as f_out:
        for vid_id in tqdm(video_folders, desc="Processing videos"):
            if vid_id not in vid_to_label:
                # Some videos might be in the subset but not in the train metadata provided
                # In a real scenario, we should handle this properly.
                continue
                
            action_text = vid_to_label[vid_id]
            action_id = label_to_id[action_text]
            
            vid_path = os.path.join(data_dir, vid_id)
            frames = sorted([img for img in os.listdir(vid_path) if img.endswith(".jpg")])
            
            # Process pairs (f_t, f_{t+stride}) with strided indices: 0, stride, 2*stride, ...
            for i in range(0, len(frames) - stride, stride):
                frame1_path = os.path.join(vid_path, frames[i])
                frame2_path = os.path.join(vid_path, frames[i + stride])
                
                latent_action = get_latent_action(laq, frame1_path, frame2_path, transform, device)
                
                record = {
                    "frame": frame1_path,
                    "next_frame": frame2_path,
                    "action": action_text,
                    "actionId": action_id,
                    "latentAction": latent_action
                }
                f_out.write(json.dumps(record) + "\n")
    
    print(f"Dataset generation complete. Saved to {output_file}")

if __name__ == "__main__":
    main()
