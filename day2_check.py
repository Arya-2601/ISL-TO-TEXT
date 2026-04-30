import os

# Updated path - dataset is on your Desktop
dataset_path = r"C:\Users\user\OneDrive\Desktop\ISL_project"

print("=== Dataset Check ===\n")

total_images = 0

for folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        count = len([f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_images += count
        print(f"  {folder.upper()}  ->  {count} images")

print(f"\nTotal folders : 26")
print(f"Total images  : {total_images}")
print("\nDataset looks good!" if total_images > 0 else "\nNo images found, check path")