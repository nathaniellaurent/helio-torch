import torch
import numpy as np
import matplotlib.pyplot as plt
from sunpy.map import Map
from sunpy.data.sample import AIA_193_JUN2012, STEREO_A_195_JUN2012
from reproject_sun import reproject_solar_image_hg

def test_reproject_stereo_to_aia():
    # Load SunPy sample images
    aia_map = Map(AIA_193_JUN2012)
    stereo_map = Map(STEREO_A_195_JUN2012)

    # --- SunPy Map reproject_to for comparison (STEREO -> AIA) ---
    reprojected_sunpy_map = stereo_map.reproject_to(aia_map.wcs)
    reprojected_sunpy = reprojected_sunpy_map.data

    # Extract required parameters from FITS headers
    # STEREO (old observer)
    cdelt1_old = float(stereo_map.meta.get('CDELT1', 1.6))  # arcsec/pixel
    cdelt2_old = float(stereo_map.meta.get('CDELT2', cdelt1_old))
    d_sun_old = float(stereo_map.meta.get('DSUN_OBS', stereo_map.meta.get('DSUN', 1.496e8))) / 1e3  # m to km
    hl_lat_old = float(stereo_map.meta.get('HGLT_OBS', stereo_map.meta.get('CRLT_OBS', 0.0)))
    hl_lon_old = float(stereo_map.meta.get('HGLN_OBS', stereo_map.meta.get('CRLN_OBS', 0.0)))
    # AIA (new observer)
    cdelt1_new = float(aia_map.meta.get('CDELT1', 0.6))
    cdelt2_new = float(aia_map.meta.get('CDELT2', cdelt1_new))
    d_sun_new = float(aia_map.meta.get('DSUN_OBS', aia_map.meta.get('DSUN', 1.496e8))) / 1e3
    hl_lat_new = float(aia_map.meta.get('HGLT_OBS', aia_map.meta.get('CRLT_OBS', 0.0)))
    hl_lon_new = float(aia_map.meta.get('HGLN_OBS', aia_map.meta.get('CRLN_OBS', 0.0)))

    print("[DEBUG] FITS parameters:")
    print(f"  cdelt1_old (STEREO): {cdelt1_old}")
    print(f"  cdelt2_old (STEREO): {cdelt2_old}")
    print(f"  d_sun_old (STEREO): {d_sun_old}")
    print(f"  hl_lat_old (STEREO): {hl_lat_old}")
    print(f"  hl_lon_old (STEREO): {hl_lon_old}")
    print(f"  cdelt1_new (AIA): {cdelt1_new}")
    print(f"  cdelt2_new (AIA): {cdelt2_new}")
    print(f"  d_sun_new (AIA): {d_sun_new}")
    print(f"  hl_lat_new (AIA): {hl_lat_new}")
    print(f"  hl_lon_new (AIA): {hl_lon_new}")

    # Reproject STEREO to AIA's heliographic grid
    reprojected_stereo = reproject_solar_image_hg(
        torch.from_numpy(stereo_map.data),
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
        output_shape=aia_map.data.shape
    ).cpu().numpy()

    print("Reprojected STEREO->AIA shape:", reprojected_stereo.shape)

    # Quantitative error: mask invalids, compute MSE between SunPy and torch reprojection
    valid_mask = np.isfinite(reprojected_stereo) & np.isfinite(reprojected_sunpy)
    mse = np.mean((reprojected_stereo[valid_mask] - reprojected_sunpy[valid_mask]) ** 2)
    print(f"[STEREO->AIA] MSE (torch vs SunPy): {mse:.4g}")

    # Visualization: original STEREO, AIA, torch reprojection, SunPy reprojection, and difference
    from matplotlib.colors import Normalize
    import matplotlib as mpl
    def robust_norm(data, vmin_pct=1, vmax_pct=99):
        finite = np.isfinite(data)
        if not np.any(finite):
            return None, None
        vmin = np.percentile(data[finite], vmin_pct)
        vmax = np.percentile(data[finite], vmax_pct)
        return vmin, vmax

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    vmin_aia, vmax_aia = robust_norm(aia_map.data)
    vmin_stereo, vmax_stereo = robust_norm(stereo_map.data)
    vmin_torch, vmax_torch = robust_norm(reprojected_stereo)
    vmin_sunpy, vmax_sunpy = robust_norm(reprojected_sunpy)

    # Top left: STEREO original
    im0 = axes[0, 0].imshow(stereo_map.data, cmap='viridis', vmin=vmin_stereo, vmax=vmax_stereo)
    axes[0, 0].set_title('Original STEREO')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    # Top middle: AIA original
    im1 = axes[0, 1].imshow(aia_map.data, cmap='magma', vmin=vmin_aia, vmax=vmax_aia)
    axes[0, 1].set_title('Original AIA')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    # Top right: SunPy Map reproject_to (STEREO->AIA)
    im2 = axes[0, 2].imshow(reprojected_sunpy, cmap='magma', vmin=vmin_aia, vmax=vmax_aia)
    axes[0, 2].set_title('STEREO->AIA (SunPy reproject_to)')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    # Bottom left: STEREO->AIA (torch)
    reprojected_masked = np.ma.masked_invalid(reprojected_stereo)
    im3 = axes[1, 0].imshow(reprojected_masked, cmap='magma', vmin=vmin_aia, vmax=vmax_aia)
    axes[1, 0].set_title('STEREO->AIA (torch)')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
   
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_reproject_stereo_to_aia()
