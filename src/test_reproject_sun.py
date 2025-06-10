
import os
import torch
from sunpy.map import Map
import numpy as np
import matplotlib.pyplot as plt

# Import the function to test (assume it is defined in reproject_sun.py)
from reproject_sun import reproject_solar_image_hg

def test_reproject_aia_to_hmi():
    # Paths to AIA and HMI images (relative to this script)
    base_dir = os.path.dirname(__file__)
    aia_path = os.path.join(base_dir, '../solar_data/aia_images/aia.lev1_euv_12s.2023-12-31T235930Z.193.image_lev1.fits')
    hmi_path = os.path.join(base_dir, '../solar_data/hmi_images/hmi.m_720s.20240101_000000_TAI.3.magnetogram.fits')



    aia_map = Map(aia_path)
    hmi_map = Map(hmi_path)


    # Downscale both images to 512x512 for speed
    from sunpy.image.resample import resample
    target_shape = (512, 512)
    orig_shape_aia = aia_map.data.shape
    orig_shape_hmi = hmi_map.data.shape
    aia_map = aia_map.__class__(resample(aia_map.data, target_shape), aia_map.meta)
    hmi_map = hmi_map.__class__(resample(hmi_map.data, target_shape), hmi_map.meta)

    # Extract required parameters from FITS headers
    # AIA (old observer)
    cdelt1_orig = float(aia_map.meta.get('CDELT1', 0.6))  # arcsec/pixel
    d_sun_old = float(aia_map.meta.get('DSUN_OBS', aia_map.meta.get('DSUN', 1.496e8))) / 1e3  # m to km
    hl_lat_old = float(aia_map.meta.get('HGLT_OBS', aia_map.meta.get('CRLT_OBS', 0.0)))
    hl_lon_old = float(aia_map.meta.get('HGLN_OBS', aia_map.meta.get('CRLN_OBS', 0.0)))
    # HMI (new observer)
    d_sun_new = float(hmi_map.meta.get('DSUN_OBS', hmi_map.meta.get('DSUN', 1.496e8))) / 1e3
    hl_lat_new = float(hmi_map.meta.get('HGLT_OBS', hmi_map.meta.get('CRLT_OBS', 0.0)))
    hl_lon_new = float(hmi_map.meta.get('HGLN_OBS', hmi_map.meta.get('CRLN_OBS', 0.0)))

    # Adjust cdelt for downscaling
    scale_factor_aia = orig_shape_aia[0] / target_shape[0]
    scale_factor_hmi = orig_shape_hmi[0] / target_shape[0]
    cdelt1_old = cdelt1_orig * scale_factor_aia
    cdelt2_old = float(aia_map.meta.get('CDELT2', cdelt1_orig)) * scale_factor_aia
    cdelt1_hmi_orig = float(hmi_map.meta.get('CDELT1', 0.5))  # arcsec/pixel
    cdelt2_hmi_orig = float(hmi_map.meta.get('CDELT2', cdelt1_hmi_orig))
    cdelt1_new = cdelt1_hmi_orig * scale_factor_hmi
    cdelt2_new = cdelt2_hmi_orig * scale_factor_hmi

    print("[DEBUG] FITS parameters:")
    print(f"  cdelt1_orig: {cdelt1_orig}")
    print(f"  cdelt1_old (downscaled): {cdelt1_old}")
    print(f"  cdelt2_old (downscaled): {cdelt2_old}")
    print(f"  d_sun_old: {d_sun_old}")
    print(f"  hl_lat_old: {hl_lat_old}")
    print(f"  hl_lon_old: {hl_lon_old}")
    print(f"  d_sun_new: {d_sun_new}")
    print(f"  hl_lat_new: {hl_lat_new}")
    print(f"  hl_lon_new: {hl_lon_new}")
    print(f"  cdelt1_new (output pixel scale): {cdelt1_new}")
    print(f"  cdelt2_new (output pixel scale): {cdelt2_new}")

    # Reproject AIA to HMI's heliographic grid
    reprojected_aia = reproject_solar_image_hg(
        torch.from_numpy(aia_map.data),
        d_sun_old,
        hl_lat_old,
        hl_lon_old,
        cdelt1_old,
        cdelt2_old,
        d_sun_new,
        hl_lat_new,
        hl_lon_new,
        cdelt1_new=cdelt1_new,
        cdelt2_new=cdelt2_new
    ).cpu().numpy()

    # Check output shape matches HMI
    assert reprojected_aia.shape == hmi_map.data.shape, f"Shape mismatch: {reprojected_aia.shape} vs {hmi_map.data.shape}"
    print("Reprojected AIA shape:", reprojected_aia.shape)

    # Optionally plot for visual inspection
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(aia_map.data, cmap='gray')
    axes[0].set_title('Original AIA')
    axes[1].imshow(hmi_map.data, cmap='gray')
    axes[1].set_title('Original HMI')
    axes[2].imshow(reprojected_aia, cmap='gray')
    axes[2].set_title('AIA reprojected to HMI')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_reproject_aia_to_hmi()
