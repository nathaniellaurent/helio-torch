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
    dataset = CustomImageDataset(csv_file, img_dir, 5)
    print(f"Dataset length: {len(dataset)}")
    # Try loading the first sample
    image, label = dataset[0]
    print(f"Image shape: {image.shape}")  # Should be (2, H, W)
    print(f"Label shape: {label.shape}")
  

    # Plot AIA, HMI, and overlay using tensor data
   
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # Plot AIA
    im0 = axes[0, 0].imshow(image[0])
    axes[0, 0].set_title('AIA')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    # Plot HMI
    im1 = axes[0, 1].imshow(image[1])
    axes[0, 1].set_title('HMI')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    # Overlay: show AIA as background, HMI as transparent overlay
    im2 = axes[0, 2].imshow(image[0])
    axes[0, 2].imshow(np.ma.masked_invalid(image[1]), cmap='coolwarm', alpha=0.5)
    axes[0, 2].set_title('AIA + HMI Overlay')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    # Plot label[0] (AIA-like)
    im3 = axes[1, 0].imshow(label[0])
    axes[1, 0].set_title('Label[0]')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    # Plot label[1] (HMI-like)
    im4 = axes[1, 1].imshow(label[1])
    axes[1, 1].set_title('Label[1]')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    # Overlay: show label[0] as background, label[1] as transparent overlay
    im5 = axes[1, 2].imshow(label[0])
    axes[1, 2].imshow(np.ma.masked_invalid(label[1]), cmap='coolwarm', alpha=0.5)
    axes[1, 2].set_title('Label[0] + Label[1] Overlay')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
    for row in axes:
        for ax in row:
            ax.axis('off')
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
