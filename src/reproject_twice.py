
import os
import torch
from sunpy.map import Map
import numpy as np
import matplotlib.pyplot as plt

# Import the function to test (assume it is defined in reproject_sun.py)
from reproject_sun_debug import reproject_solar_image_hg

def test_reproject_aia_to_hmi():
    # Paths to AIA and HMI images (relative to this script)
    base_dir = os.path.dirname(__file__)
    aia_path = os.path.join(base_dir, '../solar_data/aia_images/aia.lev1_euv_12s.2023-12-31T235930Z.193.image_lev1.fits')
    hmi_path = os.path.join(base_dir, '../solar_data/hmi_images/hmi.m_720s.20240101_000000_TAI.3.magnetogram.fits')



    aia_map = Map(aia_path)
    hmi_map = Map(hmi_path)

    # Print FITS header for AIA and HMI
    print("AIA FITS header:")
    print(aia_map.data)
    print(aia_map.meta)
    print("HMI FITS header:")
    print(hmi_map.data)
    print(hmi_map.meta)
    # # Downscale both images to 512x512 for speed
    # from sunpy.image.resample import resample
    # target_shape = (512, 512)
    # orig_shape_aia = aia_map.data.shape
    # orig_shape_hmi = hmi_map.data.shape
    # aia_map = aia_map.__class__(resample(aia_map.data, target_shape), aia_map.meta)
    # hmi_map = hmi_map.__class__(resample(hmi_map.data, target_shape), hmi_map.meta)

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
    hl_lon_new = -0.0137001611 + 45

    # Use original pixel scales (no downscaling)
    cdelt1_old = cdelt1_orig
    cdelt2_old = float(aia_map.meta.get('CDELT2', cdelt1_orig))
    cdelt1_hmi_orig = float(hmi_map.meta.get('CDELT1', 0.5))  # arcsec/pixel
    cdelt2_hmi_orig = float(hmi_map.meta.get('CDELT2', cdelt1_hmi_orig))
    cdelt1_new = cdelt1_hmi_orig
    cdelt2_new = cdelt2_hmi_orig

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

    # --- ROUND-TRIP REPROJECTION TEST ---
    # Step 1: Reproject AIA by +90 deg in longitude (simulate a new observer)
    hl_lon_new_90 = hl_lon_old - 90.0
    reprojected_90 = reproject_solar_image_hg(
        torch.from_numpy(aia_map.data),
        d_sun_old,
        hl_lat_old,
        hl_lon_old,
        cdelt1_old,
        cdelt2_old,
        d_sun_old,  # keep observer distance the same for round-trip
        hl_lat_old,
        hl_lon_new_90,
        cdelt1_new=cdelt1_old,
        cdelt2_new=cdelt2_old
    ).cpu().numpy()

    # Step 2: Reproject back by -90 deg (should return to original grid)
    reprojected_back = reproject_solar_image_hg(
        torch.from_numpy(reprojected_90),
        d_sun_old,
        hl_lat_old,
        hl_lon_new_90,
        cdelt1_old,
        cdelt2_old,
        d_sun_old,
        hl_lat_old,
        hl_lon_old,
        cdelt1_new=cdelt1_old,
        cdelt2_new=cdelt2_old
    ).cpu().numpy()

    # Visualization: original, after +90, after -90 (round-trip)
    from matplotlib.colors import Normalize
    import matplotlib as mpl
    def robust_norm(data, vmin_pct=1, vmax_pct=99):
        finite = np.isfinite(data)
        if not np.any(finite):
            return None, None
        vmin = np.percentile(data[finite], vmin_pct)
        vmax = np.percentile(data[finite], vmax_pct)
        return vmin, vmax

    vmin_aia, vmax_aia = robust_norm(aia_map.data)
    vmin_90, vmax_90 = robust_norm(reprojected_90)
    vmin_back, vmax_back = robust_norm(reprojected_back)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im0 = axes[0].imshow(aia_map.data, cmap='magma', vmin=vmin_aia, vmax=vmax_aia)
    axes[0].set_title('Original AIA')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.ma.masked_invalid(reprojected_90), cmap='magma', vmin=vmin_90, vmax=vmax_90)
    axes[1].set_title('AIA reprojected +90°')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(np.ma.masked_invalid(reprojected_back), cmap='magma', vmin=vmin_back, vmax=vmax_back)
    axes[2].set_title('AIA round-trip (-90°)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Optionally, print a quantitative error metric for round-trip
    diff = reprojected_back - aia_map.data
    valid = np.isfinite(diff) & np.isfinite(aia_map.data)
    if np.any(valid):
        mse = np.nanmean((diff[valid])**2)
        print(f"[Round-trip] MSE between original and round-trip: {mse:.3e}")
    else:
        print("[Round-trip] No valid pixels for error calculation.")

if __name__ == "__main__":
    test_reproject_aia_to_hmi()
