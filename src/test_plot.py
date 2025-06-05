

import os
from sunpy.map import Map
import matplotlib.pyplot as plt
from sunpy.image.resample import resample


# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script location
aia_image_path = os.path.join(script_dir, '../solar_data/aia_images/aia.lev1_euv_12s.2023-12-31T235930Z.193.image_lev1.fits')
hmi_image_path = os.path.join(script_dir, '../solar_data/hmi_images/hmi.m_720s.20240101_000000_TAI.3.magnetogram.fits')

# Load maps
aia_map = Map(aia_image_path)
hmi_map = Map(hmi_image_path)

hmi_map = hmi_map.reproject_to(aia_map.wcs)
print("Reprojection complete.")

# Create a figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': aia_map.wcs})

# Plot AIA data
aia_map.plot(axes=axes[0], title="AIA")
axes[0].set_title("AIA")

# Plot HMI data
hmi_map.plot(axes=axes[1], title="HMI")
axes[1].set_title("HMI")

# Overlay AIA and HMI on the third subplot
aia_map.plot(axes=axes[2], title="AIA + HMI Overlay")
hmi_map.plot(axes=axes[2], alpha=0.5)
axes[2].set_title("AIA + HMI Overlay")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()  