import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from custom_dataset import CustomImageDataset

# Path to the CSV and data directory
csv_file = os.path.join(os.path.dirname(__file__), '../solar_data/data_index.csv')
img_dir = os.path.join(os.path.dirname(__file__), '../solar_data')

def main():
    import matplotlib.pyplot as plt
    dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir)
    print(f"Dataset length: {len(dataset)}")
    # Try loading the first sample
    image, label = dataset[0]
    print(f"Image shape: {image.shape}")  # Should be (2, H, W)
    print(f"Label: {label}")
    # Check for NaNs or infs
    print(f"Image contains NaN: {torch.isnan(torch.tensor(image)).any()}")
    print(f"Image contains inf: {torch.isinf(torch.tensor(image)).any()}")

    # Plot AIA, HMI, and overlay using tensor data
   
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Plot AIA
    im0 = axes[0].imshow(image[0])
    axes[0].set_title('AIA')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    # Plot HMI
    im1 = axes[1].imshow(image[1])
    axes[1].set_title('HMI')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    # Overlay: show AIA as background, HMI as transparent overlay
    im2 = axes[2].imshow(image[0])
    axes[2].imshow(np.ma.masked_invalid(image[1]), cmap='coolwarm', alpha=0.5)
    axes[2].set_title('AIA + HMI Overlay')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
