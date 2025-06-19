import numpy as np
import matplotlib.pyplot as plt

def plot_sphere_projection_with_latlon(
    n_lat=9, n_lon=13, 
    sphere_radius=1.0, 
    projection='orthographic',
    title='Sphere Projection with Latitude/Longitude'
):
    """
    Plot a 2D projection of a sphere with labeled longitude and latitude lines.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    # Draw the sphere outline
    circle = plt.Circle((0, 0), sphere_radius, color='k', fill=False, lw=2)
    ax.add_artist(circle)

    # Latitude lines (from -80 to +80 deg, every 20 deg)
    latitudes = np.array([-80, -60, -40, -20, 0, 20, 40, 60, 80])
    longitudes = np.array([ -90, -60,-45, -30, 0, 30,45, 60, 90])

    # Draw latitude lines
    for lat in latitudes:
        lat_rad = np.deg2rad(lat)
        phi = np.linspace(-180, 180, 361)
        phi_rad = np.deg2rad(phi)
        x = sphere_radius * np.cos(lat_rad) * np.sin(phi_rad)
        y = sphere_radius * np.sin(lat_rad) * np.ones_like(phi_rad)
        ax.plot(x, y, color='b', lw=0.8, alpha=0.7)
        # Label latitude at phi=0
        if lat != 0:
            ax.text(0, sphere_radius * np.sin(lat_rad), f'{lat:.0f}°', color='b',
                    ha='left' if lat > 0 else 'right', va='center', fontsize=9)

    # Draw longitude lines
    for lon in longitudes:
        lon_rad = np.deg2rad(lon)
        theta = np.linspace(-90, 90, 181)
        theta_rad = np.deg2rad(theta)
        x = sphere_radius * np.cos(theta_rad) * np.sin(lon_rad)
        y = sphere_radius * np.sin(theta_rad)
        ax.plot(x, y, color='r', lw=0.8, alpha=0.7)
        # Label longitude at equator, skip ±180° and only label every label_step
        if lon != 0 and -180 < lon < 180:
            ax.text(sphere_radius * np.sin(lon_rad), 0, f'{lon:.0f}°', color='r',
                    ha='center', va='bottom', fontsize=9)

    # Center labels
    ax.text(0, 0, '0°', color='k', ha='center', va='center', fontsize=10, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xlim(-sphere_radius*1.1, sphere_radius*1.1)
    ax.set_ylim(-sphere_radius*1.1, sphere_radius*1.1)
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    # Instead of plt.show(), return the image as a numpy array
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = img[..., :3]  # Drop alpha channel for RGB
    plt.close(fig)
    return img
