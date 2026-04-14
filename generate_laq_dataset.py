import json
import os

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

if __name__ == "__main__":
    vid_to_label, label_to_id = load_metadata()
    print(f"Loaded {len(vid_to_label)} video mappings and {len(label_to_id)} action categories.")
