import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

from typing import Optional



def heliographic_to_cartesian(d_sun, lat_deg, lon_deg):
    """
    Convert observer heliographic latitude & longitude (degrees) to Cartesian coordinates.
    """
    lat = torch.deg2rad(lat_deg.double())
    lon = torch.deg2rad(lon_deg.double())
    x = d_sun.double() * torch.cos(lat) * torch.sin(lon)
    y = d_sun.double() * torch.sin(lat)
    z = d_sun.double() * torch.cos(lat) * torch.cos(lon)
    return torch.stack([x, y, z])

def reproject_solar_image_hg(

    
    image: torch.Tensor,
    d_sun_old: float,
    hl_lat_old: float,
    hl_lon_old: float,
    cdelt1_old: float,
    cdelt2_old: float,
    d_sun_new: float,
    hl_lat_new: float,
    hl_lon_new: float,
    cdelt1_new: float = None,
    cdelt2_new: float = None,
    solar_radius_km: float = 6.9634e5,
    output_shape: Optional[tuple] = None,
    crpix1_old: Optional[float] = None,
    crpix2_old: Optional[float] = None,
    crpix1_new: Optional[float] = None,
    crpix2_new: Optional[float] = None,
    crota2_old: Optional[float] = None,
    crota2_new: Optional[float] = None
) -> torch.Tensor:
    """
    Reproject a helioprojective solar image to a new observer position defined by heliographic coords.

    Parameters:
        image (torch.Tensor): 2D image tensor (H, W).
        cdelt1 (float): Pixel scale in arcsec/pixel.
        d_sun_old (float): Distance to the Sun from original observer in km.
        hl_lat_old (float): Original observer heliographic latitude in degrees.
        hl_lon_old (float): Original observer heliographic longitude in degrees.
        d_sun_new (float): New observer distance to Sun in km.
        hl_lat_new (float): New observer latitude in degrees.
        hl_lon_new (float): New observer longitude in degrees.
        solar_radius_km (float): Solar radius in km.

    Returns:
        torch.Tensor: Reprojected image as seen from new observer.
    """
    device: torch.device = image.device
    image = image.double()

    if output_shape is not None:
        H, W = output_shape
    else:
        H, W = image.shape

    H_old, W_old = image.shape

    # Set reference pixel (center of Sun) for input and output
    # FITS convention: CRPIX1/2 are 1-based, so subtract 1 for 0-based Python
    if crpix1_old is not None:
        cx = crpix1_old - 1
    else:
        cx = W // 2
    if crpix2_old is not None:
        cy = crpix2_old - 1
    else:
        cy = H // 2

    if crpix1_new is not None:
        cx_new = crpix1_new - 1
    else:
        cx_new = W // 2
    if crpix2_new is not None:
        cy_new = crpix2_new - 1
    else:
        cy_new = H // 2


    # Use a single pixel scale for input and output (arcsec/pixel)
  
    if cdelt1_new is None:
        cdelt1_new = cdelt1_old
    if cdelt2_new is None:
        cdelt2_new = cdelt2_old
    scale_rad1_old = math.radians(cdelt1_old / 3600)
    scale_rad2_old = math.radians(cdelt2_old / 3600)
    scale_rad1_new = math.radians(cdelt1_new / 3600)
    scale_rad2_new = math.radians(cdelt2_new / 3600)
    

    # Grid of helioprojective angles for input image
    y, x = torch.meshgrid(
        (torch.arange(H, device=device, dtype=torch.float64) - cy_new),
        (torch.arange(W, device=device, dtype=torch.float64) - cx_new),
        indexing='ij'
    )
    theta_x = x * scale_rad1_new
    theta_y = y * scale_rad2_new
    



    # Calculate radial angle theta_r and its sine
    theta_r = torch.sqrt(theta_x ** 2 + theta_y ** 2)
    sin_theta_r = torch.sin(theta_r)

    # Distance from observer to feature along the line of sight for each pixel using the law of sines
    
    sin_alpha = (d_sun_new / solar_radius_km) * sin_theta_r
    alpha = torch.arcsin(sin_alpha)

    alpha = math.radians(180) - alpha
    phi = math.radians(180) - alpha - theta_r
    sin_phi = torch.sin(phi)

    # distance_to_feature = torch.full_like(theta_r, float('nan'))
    distance_to_feature = solar_radius_km * sin_phi / sin_theta_r
    # Convert to heliocentric cartesian coordinates (x, y, z) for each pixel
    x_heliocentric = distance_to_feature * torch.cos(theta_y) * torch.sin(theta_x)
    y_heliocentric = distance_to_feature * torch.sin(theta_y)
    z_heliocentric = d_sun_new - distance_to_feature * torch.cos(theta_y) * torch.cos(theta_x)

    # Stack into a 3D array for further processing if needed
    pts_helio = torch.stack([x_heliocentric, y_heliocentric, z_heliocentric], dim=-1)  # shape (H, W, 3)



    # --- Rotate the 3D points by the difference in observer longitude and latitude, then by crota2_new-crota2_old about z ---
    dlat = math.radians(hl_lat_new - hl_lat_old)
    dlon = math.radians(hl_lon_new - hl_lon_old)

    # Rotation matrix for latitude (about x-axis)
    Rx = torch.tensor([
        [1, 0, 0],
        [0, math.cos(dlat), -math.sin(dlat)],
        [0, math.sin(dlat), math.cos(dlat)]
    ], dtype=torch.float64, device=device)

    # Rotation matrix for longitude (about y-axis)
    Ry = torch.tensor([
        [math.cos(dlon), 0, math.sin(dlon)],
        [0, 1, 0],
        [-math.sin(dlon), 0, math.cos(dlon)]
    ], dtype=torch.float64, device=device)

    # Rotation about z-axis by CROTA2 


    Rz1 = torch.eye(3, dtype=torch.float64, device=device)
    Rz2 = torch.eye(3, dtype=torch.float64, device=device)

    if crota2_new is not None:
        crota2_new_rad = math.radians(crota2_new)
        Rz1 = torch.tensor([
            [math.cos(-crota2_new_rad), -math.sin(-crota2_new_rad), 0],
            [math.sin(-crota2_new_rad),  math.cos(-crota2_new_rad), 0],
            [0, 0, 1]
        ], dtype=torch.float64, device=device)

    if crota2_old is not None:
        crota2_old_rad = math.radians(crota2_old)
        Rz2 = torch.tensor([
            [math.cos(crota2_old_rad), -math.sin(crota2_old_rad), 0],
            [math.sin(crota2_old_rad),  math.cos(crota2_old_rad), 0],
            [0, 0, 1]
        ], dtype=torch.float64, device=device)
        # Apply Rz1 after Ry and Rx and before Rz2
        
    R = Rz2 @ Ry @ Rx @ Rz1
    

    # Apply rotation to all points (H, W, 3)
    pts_helio_rot = torch.einsum('ij,hwj->hwi', R, pts_helio)

  
    x_heliocentric, y_heliocentric, z_heliocentric = pts_helio_rot[..., 0], pts_helio_rot[..., 1], pts_helio_rot[..., 2]

    

    # Mark pixels as invalid if z_heliocentric < 0 (i.e., behind the Sun)
    valid = z_heliocentric >= 0
    
    
    # --- Convert to helioprojective cartesian from new observer perspective ---
    # d_proj: distance from new observer to each point
    obs_distance_to_feature = torch.sqrt(x_heliocentric**2 + y_heliocentric**2 + (d_sun_old - z_heliocentric)**2)
    # theta_x_new: arg(D - z, x) = arctan2(x, D - z)
    theta_x_new = torch.atan2(x_heliocentric, d_sun_old - z_heliocentric)
    # theta_y_new: arcsin(y / d_proj)
    theta_y_new = torch.asin(y_heliocentric / obs_distance_to_feature)


    x_new = theta_x_new / scale_rad1_old + cx
    y_new = theta_y_new / scale_rad2_old + cy

    # Normalize for grid_sample
    x_norm = 2 * (x_new / (W_old - 1)) - 1
    y_norm = 2 * (y_new / (H_old - 1)) - 1


    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
    grid = grid.to(torch.float64)  # Ensure grid is double
    

    # Interpolate using bilinear sampling
    image_reshaped = image.unsqueeze(0).unsqueeze(0)
    grid[~valid.unsqueeze(0)] = float('nan')
    # grid_sample only supports float32, so cast to float32 for sampling
    warped = F.grid_sample(image_reshaped.float(), grid.float(), align_corners=True, padding_mode='zeros')



    print(f"[DEBUG] Image shape: {image.shape}, device: {device}")


    print(f"[DEBUG] Input pixel scale (radians): cdelt1={scale_rad1_old}, cdelt2={scale_rad2_old}")
    print(f"[DEBUG] Output pixel scale (radians): cdelt1={scale_rad1_new}, cdelt2={scale_rad2_new}")

    print(f"[DEBUG] theta_x/theta_y shapes: {theta_x.shape}, {theta_y.shape}")
    print(f"[DEBUG] theta_x: min={theta_x.min().item()}, max={theta_x.max().item()}, mean={theta_x.mean().item()}")
    print(f"[DEBUG] theta_y: min={theta_y.min().item()}, max={theta_y.max().item()}, mean={theta_y.mean().item()}")


    heliocentric_distance = torch.sqrt(x_heliocentric**2 + y_heliocentric**2 + z_heliocentric**2)
    fig_r, ax_r = plt.subplots(figsize=(6, 5))
    im_r = ax_r.imshow(heliocentric_distance.cpu().numpy(), cmap='cividis')
    ax_r.set_title('Heliocentric distance (km)')
    plt.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # Plot theta_x and theta_y for debugging
    fig_theta, axes_theta = plt.subplots(1, 2, figsize=(12, 5))
    im_tx = axes_theta[0].imshow(theta_x.cpu().numpy(), cmap='plasma')
    axes_theta[0].set_title('theta_x')
    plt.colorbar(im_tx, ax=axes_theta[0], fraction=0.046, pad=0.04)
    im_ty = axes_theta[1].imshow(theta_y.cpu().numpy(), cmap='plasma')
    axes_theta[1].set_title('theta_y')
    plt.colorbar(im_ty, ax=axes_theta[1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    # Show valid mask for debugging
    fig_valid, ax_valid = plt.subplots(figsize=(6, 5))
    im_valid = ax_valid.imshow(valid.cpu().numpy(), cmap='gray')
    ax_valid.set_title('valid mask (z_heliocentric >= 0)')
    plt.colorbar(im_valid, ax=ax_valid, fraction=0.046, pad=0.04)
    plt.tight_layout()
    

    # Display for debugging
    fig_proj, axes_proj = plt.subplots(1, 3, figsize=(18, 5))
    im_dx = axes_proj[0].imshow(obs_distance_to_feature.cpu().numpy(), cmap='magma')
    axes_proj[0].set_title('obs_distance_to_feature (km)')
    plt.colorbar(im_dx, ax=axes_proj[0], fraction=0.046, pad=0.04)
    im_tx = axes_proj[1].imshow(theta_x_new.cpu().numpy(), cmap='plasma')
    axes_proj[1].set_title('theta_x_new (rad)')
    plt.colorbar(im_tx, ax=axes_proj[1], fraction=0.046, pad=0.04)
    im_ty = axes_proj[2].imshow(theta_y_new.cpu().numpy(), cmap='plasma')
    axes_proj[2].set_title('theta_y_new (rad)')
    plt.colorbar(im_ty, ax=axes_proj[2], fraction=0.046, pad=0.04)
    plt.tight_layout()

    # Display x, y, z for debugging
    fig_xyz, axes_xyz = plt.subplots(1, 3, figsize=(18, 5))
    imx = axes_xyz[0].imshow(x_heliocentric.cpu().numpy(), cmap='coolwarm')
    axes_xyz[0].set_title('x_heliocentric (km)')
    plt.colorbar(imx, ax=axes_xyz[0], fraction=0.046, pad=0.04)
    imy = axes_xyz[1].imshow(y_heliocentric.cpu().numpy(), cmap='coolwarm')
    axes_xyz[1].set_title('y_heliocentric (km)')
    plt.colorbar(imy, ax=axes_xyz[1], fraction=0.046, pad=0.04)
    imz = axes_xyz[2].imshow(z_heliocentric.cpu().numpy(), cmap='coolwarm')
    axes_xyz[2].set_title('z_heliocentric (km)')
    plt.colorbar(imz, ax=axes_xyz[2], fraction=0.046, pad=0.04)
    plt.tight_layout()



    # Display alpha for debugging
    fig_alpha, ax_alpha = plt.subplots(figsize=(6, 5))
    im_alpha = ax_alpha.imshow(alpha.cpu().numpy(), cmap='viridis')
    ax_alpha.set_title('alpha (radians)')
    plt.colorbar(im_alpha, ax=ax_alpha, fraction=0.046, pad=0.04)
    plt.tight_layout()


    # Display distance_to_feature for debugging
    fig_dist, ax_dist = plt.subplots(figsize=(6, 5))
    im_dist = ax_dist.imshow(distance_to_feature.cpu().numpy(), cmap='magma')
    ax_dist.set_title('distance_to_feature (km)')
    plt.colorbar(im_dist, ax=ax_dist, fraction=0.046, pad=0.04)
    plt.tight_layout()
    # plt.show()



    
    print(f"[DEBUG] x_new/y_new shapes: {x_new.shape}, {y_new.shape}")


      # Display x_new and y_new for debugging
    fig_xynew, axes_xynew = plt.subplots(1, 2, figsize=(12, 5))
    im_xnew = axes_xynew[0].imshow(x_new.cpu().numpy(), cmap='viridis')
    axes_xynew[0].set_title('x_new (pixels)')
    plt.colorbar(im_xnew, ax=axes_xynew[0], fraction=0.046, pad=0.04)
    im_ynew = axes_xynew[1].imshow(y_new.cpu().numpy(), cmap='viridis')
    axes_xynew[1].set_title('y_new (pixels)')
    plt.colorbar(im_ynew, ax=axes_xynew[1], fraction=0.046, pad=0.04)
    plt.tight_layout()

   

    # Set invalid pixels to nan in x_norm for visualization
    x_norm_vis = x_norm.clone()
    x_norm_vis[~valid] = float('nan')

    # Display x_norm and y_norm for debugging
    fig_norm, axes_norm = plt.subplots(1, 2, figsize=(12, 5))
    im_xnorm = axes_norm[0].imshow(x_norm_vis.cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    axes_norm[0].set_title('x_norm (for grid_sample, invalid=nan)')
    plt.colorbar(im_xnorm, ax=axes_norm[0], fraction=0.046, pad=0.04)
    im_ynorm = axes_norm[1].imshow(y_norm.cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    axes_norm[1].set_title('y_norm (for grid_sample)')
    plt.colorbar(im_ynorm, ax=axes_norm[1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    # --- Overlay black pixels at (x_new, y_new) on the original AIA image ---
    # We'll use the original image as the background
    aia_img = image.detach().clone()
    # Robust normalization using torch
    finite_mask = torch.isfinite(aia_img)
    if finite_mask.any():
        aia_vmin = torch.quantile(aia_img[finite_mask], 0.01)
        aia_vmax = torch.quantile(aia_img[finite_mask], 0.99)
    else:
        aia_vmin = torch.tensor(0.0, device=device)
        aia_vmax = torch.tensor(1.0, device=device)
    aia_img_norm = (aia_img - aia_vmin) / (aia_vmax - aia_vmin + 1e-8)
    aia_img_norm = torch.clamp(aia_img_norm, 0, 1)
    # Convert to RGB
    aia_rgb = torch.stack([aia_img_norm]*3, dim=-1)
    # Round and clamp coordinates to valid pixel indices
    x_int = torch.round(x_new).long()
    y_int = torch.round(y_new).long()
    # Only plot for valid pixels and those within bounds
    mask = (
        valid &
        (x_int >= 0) & (x_int < W_old) &
        (y_int >= 0) & (y_int < H_old)
    )
    # Overlay colored, translucent pixels at (x_new, y_new)
    aia_rgb_overlay = aia_rgb.clone()
    # Choose a color (e.g., red) and alpha for overlay
    overlay_color = torch.tensor([1.0, 0.0, 0.0], device=device)  # Red
    alpha = 0.4  # 0=fully transparent, 1=opaque
    # For each valid (y_int, x_int), blend overlay color with background
    for c in range(3):
        aia_rgb_overlay[y_int[mask], x_int[mask], c] = (
            (1 - alpha) * aia_rgb_overlay[y_int[mask], x_int[mask], c] + alpha * overlay_color[c]
        )
    # Show the result
    fig_canvas, ax_canvas = plt.subplots(figsize=(6, 6))
    ax_canvas.imshow(aia_rgb_overlay.cpu().numpy(), vmin=0, vmax=1)
    ax_canvas.set_title('AIA with translucent red at (x_new, y_new)')
    ax_canvas.axis('off')
    plt.tight_layout()

    

    print(f"[DEBUG] grid shape: {grid.shape}, dtype: {grid.dtype}")
    print(f"[DEBUG] warped shape: {warped.shape}")
    
    return warped.squeeze(0).squeeze(0)
    