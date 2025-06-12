import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt



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
    solar_radius_km: float = 6.9634e5
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
    device = image.device
    image = image.double()
    H, W = image.shape
    cy, cx = H // 2, W // 2
    print(f"[DEBUG] Image shape: {image.shape}, device: {device}")


    # Use a single pixel scale for input and output (arcsec/pixel)
  
    if cdelt1_new is None:
        cdelt1_new = cdelt1_old
    if cdelt2_new is None:
        cdelt2_new = cdelt2_old
    scale_rad1_old = math.radians(cdelt1_old / 3600)
    scale_rad2_old = math.radians(cdelt2_old / 3600)
    scale_rad1_new = math.radians(cdelt1_new / 3600)
    scale_rad2_new = math.radians(cdelt2_new / 3600)
    print(f"[DEBUG] Input pixel scale (radians): cdelt1={scale_rad1_old}, cdelt2={scale_rad2_old}")
    print(f"[DEBUG] Output pixel scale (radians): cdelt1={scale_rad1_new}, cdelt2={scale_rad2_new}")

    # Grid of helioprojective angles for output image
    y, x = torch.meshgrid(
        (torch.arange(H, device=device, dtype=torch.float64) - cy),
        (torch.arange(W, device=device, dtype=torch.float64) - cx),
        indexing='ij'
    )
    theta_x = x * scale_rad1_new
    theta_y = y * scale_rad2_new
    print(f"[DEBUG] theta_x/theta_y shapes: {theta_x.shape}, {theta_y.shape}")
    print(f"[DEBUG] theta_x: min={theta_x.min().item()}, max={theta_x.max().item()}, mean={theta_x.mean().item()}")
    print(f"[DEBUG] theta_y: min={theta_y.min().item()}, max={theta_y.max().item()}, mean={theta_y.mean().item()}")
    # Plot theta_x and theta_y for debugging
    fig_theta, axes_theta = plt.subplots(1, 2, figsize=(12, 5))
    im_tx = axes_theta[0].imshow(theta_x.cpu().numpy(), cmap='plasma')
    axes_theta[0].set_title('theta_x')
    plt.colorbar(im_tx, ax=axes_theta[0], fraction=0.046, pad=0.04)
    im_ty = axes_theta[1].imshow(theta_y.cpu().numpy(), cmap='plasma')
    axes_theta[1].set_title('theta_y')
    plt.colorbar(im_ty, ax=axes_theta[1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    sin_x = torch.sin(theta_x)
    sin_y = torch.sin(theta_y)

    # Calculate radial angle theta_r and its sine
    theta_r = torch.sqrt(theta_x ** 2 + theta_y ** 2)
    sin_theta_r = torch.sin(theta_r)

    # Distance from observer to feature along the line of sight for each pixel using the law of sines
    # sin(theta_r) / solar_radius_km = sin(phi) / distance_to_feature
    
    sin_alpha = (d_sun_old / solar_radius_km) * sin_theta_r
    alpha = torch.arcsin(sin_alpha)

    alpha = math.radians(180) - alpha
    phi = math.radians(180) - alpha - theta_r
    sin_phi = torch.sin(phi)

    # distance_to_feature = torch.full_like(theta_r, float('nan'))
    distance_to_feature = solar_radius_km * sin_phi / sin_theta_r
    # Convert to heliocentric cartesian coordinates (x, y, z) for each pixel
    # x = d * cos(theta_y) * sin(theta_x)
    # y = d * sin(theta_y)
    # z = d_sun_old - d * cos(theta_y) * cos(theta_x)
    d = distance_to_feature
    x_heliocentric = d * torch.cos(theta_y) * torch.sin(theta_x)
    y_heliocentric = d * torch.sin(theta_y)
    z_heliocentric = d_sun_old - d * torch.cos(theta_y) * torch.cos(theta_x)

    # Stack into a 3D array for further processing if needed
    pts_helio = torch.stack([x_heliocentric, y_heliocentric, z_heliocentric], dim=-1)  # shape (H, W, 3)

    # --- Rotate the 3D points by the difference in observer longitude and latitude ---
    # Compute rotation angles (in radians)
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

    # Combined rotation: first latitude (Rx), then longitude (Ry)
    R = Ry @ Rx

    # Apply rotation to all points (H, W, 3)
    pts_helio_rot = torch.einsum('ij,hwj->hwi', R, pts_helio)

    

    # Calculate the new observer's heliocentric cartesian coordinates
    obs_cartesian_new = heliographic_to_cartesian(
        torch.tensor(d_sun_new, dtype=torch.float64, device=device),
        torch.tensor(hl_lat_new, dtype=torch.float64, device=device),
        torch.tensor(hl_lon_new, dtype=torch.float64, device=device)
    )
    print(f"[DEBUG] New observer cartesian coordinates: {obs_cartesian_new.cpu().numpy()}")
  
    x_heliocentric, y_heliocentric, z_heliocentric = pts_helio_rot[..., 0], pts_helio_rot[..., 1], pts_helio_rot[..., 2]

    # Mark pixels as invalid if z_heliocentric < 0 (i.e., behind the Sun)
    valid = z_heliocentric >= 0
    
    
    # --- Convert to helioprojective cartesian from new observer perspective ---
    # d_proj: distance from new observer to each point
    d_proj = torch.sqrt(x_heliocentric**2 + y_heliocentric**2 + (d_sun_new - z_heliocentric)**2)
    # theta_x_new: arg(D - z, x) = arctan2(x, D - z)
    theta_x_new = torch.atan2(x_heliocentric, d_sun_new - z_heliocentric)
    # theta_y_new: arcsin(y / d_proj)
    theta_y_new = torch.asin(y_heliocentric / d_proj)

    #    # --- Create valid mask for where NaNs would be created in theta_x_new, theta_y_new, d_proj ---
    # # d_proj must be > 0, |y_heliocentric/d_proj| <= 1
    # valid = (
    #     (d_proj > 0) &
    #     torch.isfinite(d_proj) &
    #     torch.isfinite(x_heliocentric) & torch.isfinite(y_heliocentric) & torch.isfinite(z_heliocentric) &
    #     (torch.abs(y_heliocentric / d_proj) <= 1)
    # )
    # # Optionally, mask out any points where theta_x_new or theta_y_new would be nan
    # # (e.g., due to division by zero or invalid arcsin)
    # print(f"[DEBUG] valid mask: {valid.sum().item()} / {valid.numel()} pixels valid")

    # Convert theta_x_new and theta_y_new to pixel coordinates


    # Display for debugging
    fig_proj, axes_proj = plt.subplots(1, 3, figsize=(18, 5))
    im_dx = axes_proj[0].imshow(d_proj.cpu().numpy(), cmap='magma')
    axes_proj[0].set_title('d_proj (km)')
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



    x_new = theta_x_new / scale_rad1_old + cx
    y_new = theta_y_new / scale_rad2_old + cy
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

    # Normalize for grid_sample
    x_norm = 2 * (x_new / (W - 1)) - 1
    y_norm = 2 * (y_new / (H - 1)) - 1
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
    grid = grid.to(torch.float64)  # Ensure grid is double
    print(f"[DEBUG] grid shape: {grid.shape}, dtype: {grid.dtype}")

    # Interpolate using bilinear sampling
    image_reshaped = image.unsqueeze(0).unsqueeze(0)
    grid[~valid.unsqueeze(0)] = float('nan')
    # grid_sample only supports float32, so cast to float32 for sampling
    warped = F.grid_sample(image_reshaped.float(), grid.float(), align_corners=True, padding_mode='zeros')
    print(f"[DEBUG] warped shape: {warped.shape}")
    
    return warped.squeeze(0).squeeze(0)
    