
import os
import torch
from sunpy.map import Map
import numpy as np
import matplotlib.pyplot as plt
from plot_sphere_latlon import plot_sphere_projection_with_latlon
from skimage.transform import resize


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
    hl_lat_old = 0
    hl_lon_old = float(aia_map.meta.get('HGLN_OBS', aia_map.meta.get('CRLN_OBS', 0.0)))
    hl_lon_old = 0
    # HMI (new observer)
    d_sun_new = float(hmi_map.meta.get('DSUN_OBS', hmi_map.meta.get('DSUN', 1.496e8))) / 1e3
    hl_lat_new = float(hmi_map.meta.get('HGLT_OBS', hmi_map.meta.get('CRLT_OBS', 0.0)))
    hl_lat_new = 60
    hl_lon_new = float(hmi_map.meta.get('HGLN_OBS', hmi_map.meta.get('CRLN_OBS', 0.0)))
    hl_lon_new = 90

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

    # Extract CRPIX1/2 for both input (AIA) and output (HMI)
    crpix1_old = float(aia_map.meta.get('CRPIX1', aia_map.data.shape[1] / 2 + 0.5))
    crpix2_old = float(aia_map.meta.get('CRPIX2', aia_map.data.shape[0] / 2 + 0.5))
    crpix1_new = float(hmi_map.meta.get('CRPIX1', hmi_map.data.shape[1] / 2 + 0.5))
    crpix2_new = float(hmi_map.meta.get('CRPIX2', hmi_map.data.shape[0] / 2 + 0.5))
    print(f"  crpix1_old (AIA): {crpix1_old}")
    print(f"  crpix2_old (AIA): {crpix2_old}")
    print(f"  crpix1_new (HMI): {crpix1_new}")
    print(f"  crpix2_new (HMI): {crpix2_new}")

    # Reproject AIA to HMI's heliographic grid
    reprojected_aia = reproject_solar_image_hg(
        torch.from_numpy(aia_map.data),
        d_sun_old,
        hl_lat_old,
        hl_lon_old,
        cdelt1_old,
        cdelt2_old,
        d_sun_old,
        hl_lat_new,
        hl_lon_new,
        cdelt1_new=cdelt1_old,
        cdelt2_new=cdelt2_old,
        crpix1_old=crpix1_old,
        crpix2_old=crpix2_old,
        # crpix1_new=crpix1_new,
        # crpix2_new=crpix2_new
        crota2_new=0,
        crota2_old=0
    ).cpu().numpy()

    # Check output shape matches HMI
    assert reprojected_aia.shape == hmi_map.data.shape, f"Shape mismatch: {reprojected_aia.shape} vs {hmi_map.data.shape}"
    print("Reprojected AIA shape:", reprojected_aia.shape)

    # Improved visualization: robust normalization, perceptually uniform colormaps, colorbars
    from matplotlib.colors import Normalize
    import matplotlib as mpl
    def robust_norm(data, vmin_pct=1, vmax_pct=99):
        finite = np.isfinite(data)
        if not np.any(finite):
            return None, None
        vmin = np.percentile(data[finite], vmin_pct)
        vmax = np.percentile(data[finite], vmax_pct)
        return vmin, vmax

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    # Use the same colormap and normalization for original AIA and reprojected AIA
    vmin_aia, vmax_aia = robust_norm(aia_map.data)
    # Panel 0: AIA original
    im0 = axes[0].imshow(aia_map.data, cmap='magma', vmin=vmin_aia, vmax=vmax_aia)
    axes[0].set_title('Original AIA')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    # Panel 1: HMI original (keep its own colormap)
    vmin1, vmax1 = robust_norm(hmi_map.data)
    if np.nanmin(hmi_map.data) < 0:
        vmax = np.nanpercentile(np.abs(hmi_map.data), 99)
        norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im1 = axes[1].imshow(hmi_map.data, cmap='coolwarm', norm=norm)
    else:
        im1 = axes[1].imshow(hmi_map.data, cmap='viridis', vmin=vmin1, vmax=vmax1)
    axes[1].set_title('Original HMI')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    # Panel 2: Reprojected AIA (same colormap and limits as original AIA)
    reprojected_masked = np.ma.masked_invalid(reprojected_aia)
    im2 = axes[2].imshow(reprojected_masked, cmap='magma', vmin=vmin_aia, vmax=vmax_aia)
    axes[2].set_title('AIA reprojected to HMI')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    # Panel 3: Difference (AIA - reprojected)
    diff = aia_map.data - reprojected_aia
    vmin_diff, vmax_diff = robust_norm(diff)
    im3 = axes[3].imshow(diff, cmap='seismic', vmin=vmin_diff, vmax=vmax_diff)
    axes[3].set_title('AIA - Reprojected AIA')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    # Panel 4: Overlay in different colors
    # Normalize both images to [0, 1] for RGB overlay
    aia_norm = (aia_map.data - vmin_aia) / (vmax_aia - vmin_aia + 1e-8)
    aia_norm = np.clip(aia_norm, 0, 1)
    reproj_norm = (reprojected_aia - vmin_aia) / (vmax_aia - vmin_aia + 1e-8)
    reproj_norm = np.clip(reproj_norm, 0, 1)
    # Red for original, green for reprojected
    rgb_overlay = np.zeros(aia_map.data.shape + (3,), dtype=np.float32)
    rgb_overlay[..., 0] = aia_norm  # Red channel: original
    rgb_overlay[..., 1] = reproj_norm  # Green channel: reprojected
    # Optionally, blue channel could be used for another image
    axes[4].imshow(rgb_overlay)
    axes[4].set_title('Overlay: Red=Original, Green=Reprojected')
    axes[4].axis('off')
    for ax in axes[:4]:
        ax.axis('off')
    plt.tight_layout()
    # Get the sphere overlay as an image (RGB numpy array)
    sphere_img = plot_sphere_projection_with_latlon()

    # Resample the sphere image to 4096x4096 for overlay
    sphere_img_resampled = resize(sphere_img, (4096, 4096, 3), order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)

    # Show the reprojected AIA image with the sphere overlay
    # We'll use the same robust normalization as above
    vmin_aia, vmax_aia = None, None
    try:
        vmin_aia, vmax_aia = robust_norm(reprojected_aia)
    except Exception:
        vmin_aia, vmax_aia = np.nanmin(reprojected_aia), np.nanmax(reprojected_aia)

    fig, ax = plt.subplots(figsize=(7, 7))
    # Display the reprojected AIA image in grayscale
    ax.imshow(reprojected_aia, cmap='gray', vmin=vmin_aia, vmax=vmax_aia)
    # Overlay the resampled sphere image with some transparency
    ax.imshow(sphere_img_resampled, alpha=0.5)
    ax.set_title('AIA reprojected to HMI with Sphere Lat/Lon Overlay')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_reproject_aia_to_hmi()

