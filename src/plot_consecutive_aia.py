import os
import matplotlib.pyplot as plt
import numpy as np
from sunpy.map import Map
import pandas as pd

# Path to the CSV and data directory
csv_file = os.path.join(os.path.dirname(__file__), '../solar_data/data_index.csv')
img_dir = os.path.join(os.path.dirname(__file__), '../solar_data')

def main():
    df = pd.read_csv(csv_file)
    n = 5
    step = 10
    fig, axes = plt.subplots(1, n, figsize=(2*n, 3))
    for i in range(n):
        idx = i * step
        if idx >= len(df):
            break
        aia_img_rel = df.iloc[idx, 3]  # 4th column is AIA image path
        aia_img_path = os.path.join(img_dir, aia_img_rel)
        aia_map = Map(aia_img_path)
        aia_img = aia_map.data
        ax = axes[i]
        im = ax.imshow(aia_img, cmap='magma')
        ax.set_title(f"AIA {idx}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
