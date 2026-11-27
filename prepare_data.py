import os
import shutil
import random

def split_dataset(raw_dir, output_dir="dataset", split_ratio=0.8):
    # Map raw folder names to your exact target class names
    class_mapping = {
        "Clean": "good_condition",
        "Electrical-damage": "electric_faulty",
        "Physical-Damage": "physical_damage"
    }

    random.seed(42)

    for raw_folder, target_class in class_mapping.items():
        src_path = os.path.join(raw_dir, raw_folder)
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} not found. Check folder name.")
            continue

        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        splits = {
            "train": images[:split_idx],
            "val": images[split_idx:]
        }

        for split_name, file_list in splits.items():
            dest_dir = os.path.join(output_dir, split_name, target_class)
            os.makedirs(dest_dir, exist_ok=True)
            for fname in file_list:
                shutil.copy2(os.path.join(src_path, fname), os.path.join(dest_dir, fname))

        print(f"Processed '{target_class}': {len(splits['train'])} train, {len(splits['val'])} val")

if __name__ == "__main__":
    # Point this to the extracted folder containing Clean, Electrical-damage, Physical-Damage
    split_dataset(raw_dir="raw_solar_dataset", output_dir="dataset")