
import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


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
    cdelt1_new: Optional[float] = None,
    cdelt2_new: Optional[float] = None,
    solar_radius_km: float = 6.9634e5,
    output_shape: Optional[tuple] = None,
    crpix1_old: Optional[float] = None,
    crpix2_old: Optional[float] = None,
    crpix1_new: Optional[float] = None,
    crpix2_new: Optional[float] = None
) -> torch.Tensor:
    """
    Reproject a helioprojective solar image to a new observer position defined by heliographic coordinates.

    Parameters
    ----------
    image : torch.Tensor
        2D image tensor (H, W).
    d_sun_old : float
        Distance to the Sun from original observer in km.
    hl_lat_old : float
        Original observer heliographic latitude in degrees.
    hl_lon_old : float
        Original observer heliographic longitude in degrees.
    cdelt1_old : float
        Pixel scale in arcsec/pixel (X axis) for input image.
    cdelt2_old : float
        Pixel scale in arcsec/pixel (Y axis) for input image.
    d_sun_new : float
        New observer distance to Sun in km.
    hl_lat_new : float
        New observer latitude in degrees.
    hl_lon_new : float
        New observer longitude in degrees.
    cdelt1_new : float, optional
        Pixel scale in arcsec/pixel (X axis) for output image. Defaults to input value.
    cdelt2_new : float, optional
        Pixel scale in arcsec/pixel (Y axis) for output image. Defaults to input value.
    solar_radius_km : float, optional
        Solar radius in km. Default is 6.9634e5.
    output_shape : tuple, optional
        Output image shape as (H, W). If not specified, uses input image shape.

    Returns
    -------
    torch.Tensor
        Reprojected image as seen from new observer (2D tensor, shape = output_shape or input shape).
    """
    device: torch.device = image.device
    image = image.double()

    if output_shape is not None:
        H, W = output_shape
    else:
        H, W = image.shape

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

    # Grid of helioprojective angles for output image
    # Use output reference pixel for grid definition
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
    # sin(theta_r) / solar_radius_km = sin(phi) / distance_to_feature
    
    sin_alpha = (d_sun_old / solar_radius_km) * sin_theta_r
    alpha = torch.arcsin(sin_alpha)

    alpha = math.radians(180) - alpha
    phi = math.radians(180) - alpha - theta_r
    sin_phi = torch.sin(phi)

    distance_to_feature = solar_radius_km * sin_phi / sin_theta_r

    # Convert to heliocentric cartesian coordinates (x, y, z) for each pixel
    # x = d * cos(theta_y) * sin(theta_x)
    # y = d * sin(theta_y)
    # z = d_sun_old - d * cos(theta_y) * cos(theta_x)

    d = distance_to_feature
    x_heliocentric = d * torch.cos(theta_y) * torch.sin(theta_x)
    y_heliocentric = d * torch.sin(theta_y)
    z_heliocentric = d_sun_old - d * torch.cos(theta_y) * torch.cos(theta_x)

    # Stack into a 3D array for rotation processing
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
  
    x_heliocentric, y_heliocentric, z_heliocentric = pts_helio_rot[..., 0], pts_helio_rot[..., 1], pts_helio_rot[..., 2]

    # Mark pixels as invalid if z_heliocentric < 0 (i.e., not visible from new observer)
    valid = z_heliocentric >= 0
    
    
    # --- Convert to helioprojective cartesian from new observer perspective ---
    # d_proj: distance from new observer to each point
    d_proj = torch.sqrt(x_heliocentric**2 + y_heliocentric**2 + (d_sun_new - z_heliocentric)**2)
    # theta_x_new: arg(D - z, x) = arctan2(x, D - z)
    theta_x_new = torch.atan2(x_heliocentric, d_sun_new - z_heliocentric)
    # theta_y_new: arcsin(y / d_proj)
    theta_y_new = torch.asin(y_heliocentric / d_proj)

    x_new = theta_x_new / scale_rad1_old + cx
    y_new = theta_y_new / scale_rad2_old + cy


    # Normalize for grid_sample
    x_norm = 2 * (x_new / (W - 1)) - 1
    y_norm = 2 * (y_new / (H - 1)) - 1
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
    grid = grid.to(torch.float64)  # Ensure grid is double

    # Interpolate using bilinear sampling
    image_reshaped = image.unsqueeze(0).unsqueeze(0)
    grid[~valid.unsqueeze(0)] = float('nan')
    # grid_sample only supports float32, so cast to float32 for sampling
    warped = F.grid_sample(image_reshaped.float(), grid.float(), align_corners=False, padding_mode='zeros')
    
    return warped.squeeze(0).squeeze(0)
    