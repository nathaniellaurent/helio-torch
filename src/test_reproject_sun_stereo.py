import torch
import numpy as np
import matplotlib.pyplot as plt
from sunpy.map import Map
from sunpy.data.sample import AIA_193_JUN2012, STEREO_A_195_JUN2012
from reproject_sun_debug import reproject_solar_image_hg



def test_reproject_aia_to_stereo():
   
    
    

    aia_map = Map(AIA_193_JUN2012)
    stereo_map = Map(STEREO_A_195_JUN2012)

    # --- SunPy Map reproject_to for comparison ---
    reprojected_sunpy_map = aia_map.reproject_to(stereo_map.wcs)
    reprojected_sunpy = reprojected_sunpy_map.data

    print("[DEBUG] AIA WCS:")
    print(aia_map.wcs)
    print("[DEBUG] STEREO WCS:")
    print(stereo_map.wcs)
   
    # Load SunPy sample images
   

    # Extract required parameters from FITS headers
    # AIA (old observer)
    cdelt1_orig = float(aia_map.meta.get('CDELT1', 0.6))  # arcsec/pixel
    d_sun_old = float(aia_map.meta.get('DSUN_OBS', aia_map.meta.get('DSUN', 1.496e8))) / 1e3  # m to km
    hl_lat_old = float(aia_map.meta.get('HGLT_OBS', aia_map.meta.get('CRLT_OBS', 0.0)))
    hl_lon_old = float(aia_map.meta.get('HGLN_OBS', aia_map.meta.get('CRLN_OBS', 0.0)))
    # STEREO (new observer)
    d_sun_new = float(stereo_map.meta.get('DSUN_OBS', stereo_map.meta.get('DSUN', 1.496e8))) / 1e3
    hl_lat_new = float(stereo_map.meta.get('HGLT_OBS', stereo_map.meta.get('CRLT_OBS', 0.0)))
    hl_lon_new = float(stereo_map.meta.get('HGLN_OBS', stereo_map.meta.get('CRLN_OBS', 0.0)))

    # Use original pixel scales (no downscaling)
    cdelt1_old = cdelt1_orig
    cdelt2_old = float(aia_map.meta.get('CDELT2', cdelt1_orig))
    cdelt1_stereo_orig = float(stereo_map.meta.get('CDELT1', 1.6))  # arcsec/pixel
    cdelt2_stereo_orig = float(stereo_map.meta.get('CDELT2', cdelt1_stereo_orig))
    cdelt1_new = cdelt1_stereo_orig
    cdelt2_new = cdelt2_stereo_orig

    
    
    # Extract CRPIX1/2 for both input (AIA) and output (STEREO)
    crpix1_old = float(aia_map.meta.get('CRPIX1', aia_map.data.shape[1] / 2 + 0.5))
    crpix2_old = float(aia_map.meta.get('CRPIX2', aia_map.data.shape[0] / 2 + 0.5))
    crpix1_new = float(stereo_map.meta.get('CRPIX1', 2048 / 2 + 0.5))
    crpix2_new = float(stereo_map.meta.get('CRPIX2', 2048 / 2 + 0.5))

    # Print the FITS parameter that is the angle difference between celestial north and the north pole of the sun
    # This is usually the 'CROTA2' keyword (rotation angle of the image)
    # Compute equivalent CROTA2 for STEREO using PCi_j if present
    pc11 = float(stereo_map.meta.get('PC1_1', 1.0))
    pc12 = float(stereo_map.meta.get('PC1_2', 0.0))
    pc21 = float(stereo_map.meta.get('PC2_1', 0.0))
    pc22 = float(stereo_map.meta.get('PC2_2', 1.0))
    # Only compute if PC matrix is present or nontrivial
    if any([abs(pc11 - 1.0) > 1e-6, abs(pc12) > 1e-6, abs(pc21) > 1e-6, abs(pc22 - 1.0) > 1e-6]):
        crota2_stereo_pc = np.degrees(np.arctan2(pc21, pc11))
    else:
        crota2_stereo_pc = 0.0

    print(f"  crota2_stereo (computed from PCi_j): {crota2_stereo_pc}")

    # --- Repeat for AIA ---
    pc11_aia = float(aia_map.meta.get('PC1_1', 1.0))
    pc12_aia = float(aia_map.meta.get('PC1_2', 0.0))
    pc21_aia = float(aia_map.meta.get('PC2_1', 0.0))
    pc22_aia = float(aia_map.meta.get('PC2_2', 1.0))
    if any([abs(pc11_aia - 1.0) > 1e-6, abs(pc12_aia) > 1e-6, abs(pc21_aia) > 1e-6, abs(pc22_aia - 1.0) > 1e-6]):
        crota2_aia_pc = np.degrees(np.arctan2(pc21_aia, pc11_aia))
    else:
        crota2_aia_pc = 0.0

    print(f"  crota2_aia (computed from PCi_j): {crota2_aia_pc}")


    crota2_aia = aia_map.meta.get('CROTA2', None)
    crota2_stereo = stereo_map.meta.get('CROTA2', crota2_stereo_pc)

    print("[DEBUG] FITS parameters:")
    print(f"  cdelt1_orig: {cdelt1_orig}")
    print(f"  cdelt1_old: {cdelt1_old}")
    print(f"  cdelt2_old: {cdelt2_old}")
    print(f"  d_sun_old: {d_sun_old}")
    print(f"  hl_lat_old: {hl_lat_old}")
    print(f"  hl_lon_old: {hl_lon_old}")
    print(f"  d_sun_new: {d_sun_new}")
    print(f"  hl_lat_new: {hl_lat_new}")
    print(f"  hl_lon_new: {hl_lon_new}")
    print(f"  cdelt1_new (output pixel scale): {cdelt1_new}")
    print(f"  cdelt2_new (output pixel scale): {cdelt2_new}")
    print(f"  crpix1_old (AIA): {crpix1_old}")
    print(f"  crpix2_old (AIA): {crpix2_old}")
    print(f"  crpix1_new (STEREO): {crpix1_new}")
    print(f"  crpix2_new (STEREO): {crpix2_new}")
    print(f"  crota2_aia (AIA, angle celestial north - solar north): {crota2_aia}")
    print(f"  crota2_stereo (STEREO, angle celestial north - solar north): {crota2_stereo}")



   

    # Reproject AIA to STEREO's heliographic grid
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
        cdelt2_new=cdelt2_new,
        output_shape=(2048, 2048),
        crpix1_old=crpix1_old,
        crpix2_old=crpix2_old,
        crpix1_new=crpix1_new,
        crpix2_new=crpix2_new,
        crota2_old=15.3,
        crota2_new=-3.5
    ).cpu().numpy()

    # Check output shape matches STEREO
    # assert reprojected_aia.shape == stereo_map.data.shape, f"Shape mismatch: {reprojected_aia.shape} vs {stereo_map.data.shape}"
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

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # Use the same colormap and normalization for original AIA and reprojected AIA
    vmin_aia, vmax_aia = robust_norm(aia_map.data)
    # Top left: AIA original
    im0 = axes[0, 0].imshow(aia_map.data, cmap='magma', vmin=vmin_aia, vmax=vmax_aia)
    axes[0, 0].set_title('Original AIA')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    # Top right: STEREO original (keep its own colormap)
    vmin1, vmax1 = robust_norm(stereo_map.data)
    im1 = axes[0, 1].imshow(stereo_map.data, cmap='viridis', vmin=vmin1, vmax=vmax1)
    axes[0, 1].set_title('Original STEREO')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    # Bottom left: Reprojected AIA (same colormap and limits as original AIA)
    reprojected_masked = np.ma.masked_invalid(reprojected_aia)
    im2 = axes[1, 0].imshow(reprojected_masked, cmap='magma', vmin=vmin_aia, vmax=vmax_aia)
    axes[1, 0].set_title('AIA reprojected (torch)')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    # Bottom right: SunPy Map reproject_to
    vmin3, vmax3 = robust_norm(reprojected_sunpy)
    im3 = axes[1, 1].imshow(reprojected_sunpy, cmap='magma', vmin=vmin_aia, vmax=vmax_aia)
    axes[1, 1].set_title('AIA reprojected (SunPy reproject_to)')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for row in axes:
        for ax in row:
            ax.axis('off')
    plt.tight_layout()
    # Save the last output image (bottom right: SunPy Map reproject_to)
    fig.savefig('aia_to_stereo_comparison.png', dpi=200)

    # --- Overlay: difference and overlay between PyTorch and SunPy reprojections ---
    # Mask invalids for both
    mask_valid = np.isfinite(reprojected_aia) & np.isfinite(reprojected_sunpy)

    # Create copies for diff calculation
    reprojected_aia_copy = reprojected_aia.copy()
    reprojected_sunpy_copy = reprojected_sunpy.copy()

    mask_valid_aia = np.isfinite(reprojected_aia_copy)
    mask_valid_sunpy = np.isfinite(reprojected_sunpy_copy)

    reprojected_aia_copy[~mask_valid_aia] = 0
    reprojected_sunpy_copy[~mask_valid_sunpy] = 0
    diff = np.zeros_like(reprojected_aia)
    diff = reprojected_aia_copy- reprojected_sunpy_copy

    # Overlay: alpha blend
    # Normalize both to [0, 1] for overlay
    def norm01(x):
        finite = np.isfinite(x)
        if not np.any(finite):
            return np.zeros_like(x)
        vmin, vmax = np.percentile(x[finite], 1), np.percentile(x[finite], 99)
        xnorm = (x - vmin) / (vmax - vmin)
        xnorm = np.clip(xnorm, 0, 1)
        return xnorm
    norm_pt = norm01(reprojected_aia)
    norm_sp = norm01(reprojected_sunpy)
    # Red = PyTorch, Green = SunPy, Blue = 0
    overlay = np.zeros(reprojected_aia.shape + (3,), dtype=np.float32)
    overlay[..., 0] = norm_pt  # Red
    overlay[..., 1] = norm_sp  # Green
    overlay[..., 2] = 0
    # Mask out invalids
    overlay[~mask_valid] = 0

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
    imd = axes2[0].imshow(diff, cmap='coolwarm', vmin=-np.nanmax(np.abs(diff)), vmax=np.nanmax(np.abs(diff)))
    axes2[0].set_title('Difference (PyTorch - SunPy)')
    plt.colorbar(imd, ax=axes2[0], fraction=0.046, pad=0.04)
    axes2[0].axis('off')
    axes2[1].imshow(overlay)
    axes2[1].set_title('Overlay: Red=PyTorch, Green=SunPy')
    axes2[1].axis('off')
    plt.tight_layout()
    fig2.savefig('aia_to_stereo_overlay.png', dpi=200)
    # Save just the difference image as well
    plt.imsave('aia_to_stereo_difference.png', diff, cmap='coolwarm', vmin=-np.nanmax(np.abs(diff)), vmax=np.nanmax(np.abs(diff)))
    plt.show()

if __name__ == "__main__":
    test_reproject_aia_to_stereo()
