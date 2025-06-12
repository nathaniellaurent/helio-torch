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
    theta_x = x * scale_rad1_old
    theta_y = y * scale_rad2_old
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
    sum_sin2 = sin_x**2 + sin_y**2
    print(f"[DEBUG] sin_x: min={sin_x.min().item()}, max={sin_x.max().item()}, mean={sin_x.mean().item()}")
    print(f"[DEBUG] sin_y: min={sin_y.min().item()}, max={sin_y.max().item()}, mean={sin_y.mean().item()}")
    print(f"[DEBUG] sin_x**2 + sin_y**2: min={sum_sin2.min().item()}, max={sum_sin2.max().item()}, mean={sum_sin2.mean().item()}")
    z_comp = torch.sqrt(1 - sum_sin2)
    n_nan_z = torch.isnan(z_comp).sum().item()
    n_inf_z = torch.isinf(z_comp).sum().item()
    print(f"[DEBUG] z_comp: min={z_comp.min().item()}, max={z_comp.max().item()}, mean={z_comp.mean().item()}, nan={n_nan_z}, inf={n_inf_z}")
    v_hat = torch.stack([sin_x, sin_y, z_comp], dim=-1)  # [H, W, 3]
    n_nan_v = torch.isnan(v_hat).sum().item()
    n_inf_v = torch.isinf(v_hat).sum().item()
    print(f"[DEBUG] v_hat shape: {v_hat.shape}, nan={n_nan_v}, inf={n_inf_v}")

     # Original observer position (cartesian)
    r_obs = heliographic_to_cartesian(
        torch.tensor(d_sun_old, device=device, dtype=torch.float64),
        torch.tensor(hl_lat_old, device=device, dtype=torch.float64),
        torch.tensor(hl_lon_old, device=device, dtype=torch.float64)
    )
    print(f"[DEBUG] r_obs: {r_obs}")
    print(f"[DEBUG] |r_obs|: {torch.norm(r_obs).item()} (solar_radius_km={solar_radius_km})")


    # --- ROTATE v_hat SO THAT THE CENTER VECTOR POINTS TOWARD THE SUN CENTER (r_obs) ---
    # Center direction in original grid (should be [0,0,1])
    v_hat_center = v_hat[v_hat.shape[0]//2, v_hat.shape[1]//2]
    print(f"[DEBUG] v_hat_center (before rotation): {v_hat_center}")
    # Desired direction: from observer to Sun center (unit vector)
    r_obs_unit = -r_obs / torch.norm(r_obs)
    print(f"[DEBUG] r_obs_unit (should point to Sun center): {r_obs_unit}")
    # Compute rotation axis and angle
    axis = torch.cross(v_hat_center, r_obs_unit)
    axis_norm = torch.norm(axis)
    if axis_norm > 1e-8:
        axis = axis / axis_norm
        angle = torch.arccos(torch.clamp(torch.dot(v_hat_center, r_obs_unit), -1.0, 1.0))
        print(f"[DEBUG] Rotation axis: {axis}, angle (deg): {angle * 180 / math.pi}")
        # Rodrigues' rotation formula
        K = torch.tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], device=device)
        I = torch.eye(3, device=device)
        R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        v_hat_shape = v_hat.shape
        v_hat_flat = v_hat.reshape(-1, 3)
        v_hat_rot = torch.matmul(v_hat_flat, R.T)
        v_hat = v_hat_rot.reshape(v_hat_shape)
    else:
        print("[DEBUG] No rotation needed (center already aligned)")

    # Plot all three stacks of v_hat (x, y, z components)
    fig_vhat, axes_vhat = plt.subplots(1, 3, figsize=(18, 5))
    im_v0 = axes_vhat[0].imshow(v_hat[..., 0].cpu().numpy(), cmap='viridis')
    axes_vhat[0].set_title('v_hat[..., 0] (x component)')
    plt.colorbar(im_v0, ax=axes_vhat[0], fraction=0.046, pad=0.04)
    im_v1 = axes_vhat[1].imshow(v_hat[..., 1].cpu().numpy(), cmap='viridis')
    axes_vhat[1].set_title('v_hat[..., 1] (y component)')
    plt.colorbar(im_v1, ax=axes_vhat[1], fraction=0.046, pad=0.04)
    im_v2 = axes_vhat[2].imshow(v_hat[..., 2].cpu().numpy(), cmap='viridis')
    axes_vhat[2].set_title('v_hat[..., 2] (z component)')
    plt.colorbar(im_v2, ax=axes_vhat[2], fraction=0.046, pad=0.04)
    plt.tight_layout()

   
    # Ray-sphere intersection
    b = 2 * torch.tensordot(v_hat, r_obs, dims=([2], [0]))
    c = torch.dot(r_obs, r_obs) - torch.tensor(solar_radius_km, dtype=torch.float64, device=device) ** 2
    discriminant = b**2 - 4 * c
    print(f"[DEBUG] b: min={b.min().item()}, max={b.max().item()}, mean={b.mean().item()}")
    print(f"[DEBUG] c: {c.item() if hasattr(c, 'item') else c}")
    print(f"[DEBUG] discriminant: min={discriminant.min().item()}, max={discriminant.max().item()}, mean={discriminant.mean().item()}")
    # Plot b and discriminant for debugging
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(b.cpu().numpy(), cmap='viridis')
    axes[0].set_title('b')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(discriminant.cpu().numpy(), cmap='coolwarm')
    axes[1].set_title('discriminant')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    valid = discriminant > 0
    t = torch.full_like(b, float('nan'), dtype=torch.float64)
    t[valid] = (-b[valid] - torch.sqrt(discriminant[valid])) / 2
    print(f"[DEBUG] Ray-sphere intersection: valid pixels: {valid.sum().item()} / {valid.numel()}")

    # Display t (ray parameter) statistics and image
    if valid.any():
        print(f"[DEBUG] t: min={t[valid].min().item()}, max={t[valid].max().item()}, mean={t[valid].mean().item()}")
    else:
        print(f"[DEBUG] t: all values are NaN or invalid")
    fig_t, ax_t = plt.subplots(1, 1, figsize=(6, 5))
    im_t = ax_t.imshow(t.cpu().numpy(), cmap='coolwarm')
    ax_t.set_title('Ray parameter t')
    plt.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # 3D surface points on the Sun
    surface_pts = r_obs.view(1, 1, 3) + t.unsqueeze(-1) * v_hat
    print(f"[DEBUG] surface_pts shape: {surface_pts.shape}")
    # Display magnitude of surface points
    surface_pts_mag = torch.norm(surface_pts, dim=-1)
    print(f"[DEBUG] surface_pts |r|: min={surface_pts_mag.min().item()}, max={surface_pts_mag.max().item()}, mean={surface_pts_mag.mean().item()}")
    # Overlay the surface point magnitude on the original image for visualization
    fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(8, 7))
    # Show the original image in grayscale
    im_base = ax_overlay.imshow(image.cpu().numpy(), cmap='gray', alpha=0.7)
    # Overlay the surface point magnitude as a transparent heatmap
    im_mag = ax_overlay.imshow(surface_pts_mag.cpu().numpy(), cmap='magma', alpha=0.5)
    ax_overlay.set_title('AIA image with surface point |r| overlay')
    plt.colorbar(im_mag, ax=ax_overlay, fraction=0.046, pad=0.04, label='|r| (km)')
    plt.tight_layout()

    # New observer position (cartesian)
    r_new = heliographic_to_cartesian(
        torch.tensor(d_sun_new, device=device, dtype=torch.float64),
        torch.tensor(hl_lat_new, device=device, dtype=torch.float64),
        torch.tensor(hl_lon_new, device=device, dtype=torch.float64)
    )
    print(f"[DEBUG] r_new: {r_new}")

    # Line of sight vectors from new observer
    v_new = surface_pts - r_new.view(1, 1, 3)
    # Compute angle between r_new (from Sun center to observer) and surface_pts (from Sun center to surface)
    # r_new: shape (3,), surface_pts: (H, W, 3)
    r_new_unit = r_new / torch.norm(r_new)
    surface_pts_unit = surface_pts / torch.norm(surface_pts, dim=-1, keepdim=True)
    # Cosine of angle between r_new and surface_pts
    cos_angle = (surface_pts_unit @ r_new_unit)
    # Mask: True where angle <= 90 deg (cos >= 0), False where angle > 90 deg (cos < 0)
    valid_angle = cos_angle >= 0
    # Set v_new_hat to NaN where angle > 90 deg
    v_new_hat = v_new / torch.norm(v_new, dim=-1, keepdim=True)
    v_new_hat[~valid_angle] = float('nan')
    print(f"[DEBUG] v_new_hat shape: {v_new_hat.shape}, masked {{torch.sum(~valid_angle).item()}} points with angle > 90 deg")

 # Project back to helioprojective angles from new observer (after rotation)
    theta_x_new = torch.arcsin(v_new_hat[..., 0])
    theta_y_new = torch.arcsin(v_new_hat[..., 1])
    print(f"[DEBUG] theta_x_new/theta_y_new shapes: {theta_x_new.shape}, {theta_y_new.shape}")
    print(f"[DEBUG] theta_x_new: min={theta_x_new.min().item():.10f}, max={theta_x_new.max().item():.10f}, mean={theta_x_new.mean().item():.10f}")
    print(f"[DEBUG] theta_y_new: min={theta_y_new.min().item():.10f}, max={theta_y_new.max().item():.10f}, mean={theta_y_new.mean().item():.10f}")
    # Optionally plot theta_x_new and theta_y_new
    fig_theta_new, axes_theta_new = plt.subplots(1, 2, figsize=(12, 5))
    im_txn = axes_theta_new[0].imshow(theta_x_new.cpu().numpy(), cmap='plasma')
    axes_theta_new[0].set_title('theta_x_new')
    plt.colorbar(im_txn, ax=axes_theta_new[0], fraction=0.046, pad=0.04)
    im_tyn = axes_theta_new[1].imshow(theta_y_new.cpu().numpy(), cmap='plasma')
    axes_theta_new[1].set_title('theta_y_new')
    plt.colorbar(im_tyn, ax=axes_theta_new[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    # Project back to helioprojective angles from new observer
    # --- ROTATE v_new_hat SO THAT -r_new POINTS TO +Z ---
    # Compute rotation from -r_new to [0,0,1]
    minus_r_new = -r_new / torch.norm(r_new)
    target_z = torch.tensor([0.0, 0.0, 1.0], dtype=minus_r_new.dtype, device=minus_r_new.device)
    axis2 = torch.cross(minus_r_new, target_z)
    axis2_norm = torch.norm(axis2)
    if axis2_norm > 1e-8:
        axis2 = axis2 / axis2_norm
        angle2 = torch.arccos(torch.clamp(torch.dot(minus_r_new, target_z), -1.0, 1.0))
        print(f"[DEBUG] v_new_hat rotation axis: {axis2}, angle (deg): {angle2 * 180 / math.pi}")
        K2 = torch.tensor([[0, -axis2[2], axis2[1]],
                          [axis2[2], 0, -axis2[0]],
                          [-axis2[1], axis2[0], 0]], device=device, dtype=torch.float64)
        I2 = torch.eye(3, device=device, dtype=torch.float64)
        R2 = I2 + torch.sin(angle2) * K2 + (1 - torch.cos(angle2)) * (K2 @ K2)
        vnh_shape = v_new_hat.shape
        vnh_flat = v_new_hat.reshape(-1, 3)
        vnh_rot = torch.matmul(vnh_flat, R2.T)
        v_new_hat = vnh_rot.reshape(vnh_shape)
    else:
        print("[DEBUG] No v_new_hat rotation needed (already aligned)")

    # Project back to helioprojective angles from new observer (after rotation)
    theta_x_new = torch.arcsin(v_new_hat[..., 0])
    theta_y_new = torch.arcsin(v_new_hat[..., 1])
    print(f"[DEBUG] theta_x_new/theta_y_new shapes: {theta_x_new.shape}, {theta_y_new.shape}")
    print(f"[DEBUG] theta_x_new: min={theta_x_new.min().item():.10f}, max={theta_x_new.max().item():.10f}, mean={theta_x_new.mean().item():.10f}")
    print(f"[DEBUG] theta_y_new: min={theta_y_new.min().item():.10f}, max={theta_y_new.max().item():.10f}, mean={theta_y_new.mean().item():.10f}")
    # Optionally plot theta_x_new and theta_y_new
    fig_theta_new, axes_theta_new = plt.subplots(1, 2, figsize=(12, 5))
    im_txn = axes_theta_new[0].imshow(theta_x_new.cpu().numpy(), cmap='plasma')
    axes_theta_new[0].set_title('theta_x_new')
    plt.colorbar(im_txn, ax=axes_theta_new[0], fraction=0.046, pad=0.04)
    im_tyn = axes_theta_new[1].imshow(theta_y_new.cpu().numpy(), cmap='plasma')
    axes_theta_new[1].set_title('theta_y_new')
    plt.colorbar(im_tyn, ax=axes_theta_new[1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    # Convert to pixel coordinates

    # Map new helioprojective angles to input pixel coordinates
    x_new = theta_x_new / scale_rad1_old + cx
    y_new = theta_y_new / scale_rad2_old + cy
    print(f"[DEBUG] x_new/y_new shapes: {x_new.shape}, {y_new.shape}")

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