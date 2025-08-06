#!/usr/bin/env python3
"""
IEEE Format Figure 1: System Model & Problem Definition (MEMORY OPTIMIZED)
======================================================================

This script generates a scientifically accurate, high-level conceptual diagram 
showing the OAM-based wireless communication system with intelligent handover optimization.

Features:
- Scientifically accurate Laguerre-Gaussian beam physics
- Realistic beam width evolution for 28 GHz
- Proper 3D visualization with IEEE publication quality
- Atmospheric turbulence effects
- OAM mode handover visualization
- Professional color scheme and styling
- MEMORY OPTIMIZED: Reduced array sizes, proper cleanup, efficient plotting

Author: OAM 6G Research Team
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.art3d as art3d
import gc  # For garbage collection

# --- Configuration and Styling ---

# Define a standard, professional color palette
IEEE_COLORS = {
    'blue': '#00629B',
    'green': '#009E73',
    'red': '#D55E00',
    'orange': '#E69F00',
    'purple': '#6A00A9',
    'gray': '#555555',
    'light_gray': '#DDDDDD',
    'cyan': '#0072B2',
}

# Assign specific, distinct colors for the OAM modes being visualized
OAM_COLORS = {
    1: IEEE_COLORS['blue'],
    3: IEEE_COLORS['orange'],
    # Colors for path-only representation
    2: '#56B4E9',
    4: '#F0E442',
    5: '#CC79A7',
    6: '#D55E00'
}

def setup_ieee_style():
    """Setup IEEE publication style parameters for Matplotlib."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'lines.linewidth': 2.0,
        'axes.linewidth': 1.0
    })

# --- Scientific Modeling Functions ---

def get_lg_beam_intensity(l: int, r: np.ndarray, w_z: float, p: int = 0) -> np.ndarray:
    """
    Calculates the normalized intensity of a Laguerre-Gaussian beam.
    
    This implements the correct Laguerre-Gaussian beam formula with proper normalization:
    E_lp(r,φ,z) = C * (w0/w(z)) * (sqrt(2)*r/w(z))^|l| * L_p^|l|(2r²/w²(z)) * exp(-r²/w²(z)) * exp(ilφ)
    
    Where C = sqrt(2*p!/(π(p+|l|)!)) is the normalization factor to ensure unit power.
    
    Args:
        l: OAM topological charge (azimuthal index)
        r: Radial coordinate array
        w_z: Beam width at distance z
        p: Radial index (default 0 for fundamental radial mode)
        
    Returns:
        Normalized intensity profile |E_lp|²
    """
    from scipy.special import genlaguerre, factorial
    import math
    
    # Handle l=0 case (fundamental Gaussian)
    if l == 0 and p == 0:
        # E_00 = (w0/w(z)) * exp(-r²/w²(z))
        # For p=0, l=0, the normalization factor C = sqrt(2/π)
        normalization = math.sqrt(2/math.pi)
        amplitude = normalization * np.exp(-r**2 / w_z**2)
    else:
        # Calculate normalization factor C = sqrt(2*p!/(π(p+|l|)!))
        # Use logarithm for factorial to avoid overflow for large p or l
        if p > 10 or abs(l) > 10:
            # For large values, use Stirling's approximation
            log_p_factorial = math.lgamma(p + 1)
            log_p_l_factorial = math.lgamma(p + abs(l) + 1)
            log_normalization = 0.5 * (math.log(2) + log_p_factorial - math.log(math.pi) - log_p_l_factorial)
            normalization = math.exp(log_normalization)
        else:
            # For small values, direct calculation is safe
            p_factorial = math.factorial(p)
            p_l_factorial = math.factorial(p + abs(l))
            normalization = math.sqrt(2 * p_factorial / (math.pi * p_l_factorial))
        
        # Full Laguerre-Gaussian formula for l≠0 or p≠0
        # L_p^|l|(2r²/w²) - Generalized Laguerre polynomial
        laguerre_poly = genlaguerre(p, abs(l))
        laguerre_arg = 2 * r**2 / w_z**2
        
        # Calculate Laguerre polynomial values
        laguerre_values = laguerre_poly(laguerre_arg)
        
        # Calculate the full amplitude with normalization
        amplitude = normalization * (np.sqrt(2) * r / w_z)**abs(l) * laguerre_values * np.exp(-r**2 / w_z**2)
    
    # Return intensity (amplitude squared)
    return np.abs(amplitude)**2

def get_beam_width(z: float, w0: float = 0.05, wavelength: float = 1.07e-2) -> float:
    """
    Calculate beam width at distance z using the beam propagation equation.
    
    Args:
        z: Distance from beam waist
        w0: Beam waist radius (default 5cm for 28 GHz)
        wavelength: Wavelength (default 1.07cm for 28 GHz)
        
    Returns:
        Beam width at distance z
    """
    # Rayleigh range
    z_r = np.pi * w0**2 / wavelength
    
    # Beam width at distance z
    w_z = w0 * np.sqrt(1 + (z / z_r)**2)
    
    return w_z

# --- Memory-Optimized Drawing Functions ---

def draw_base_station(ax: Axes3D, pos: tuple):
    """Draws a base station with memory-efficient plotting."""
    x, y, z = pos
    ax.scatter(x, y, z, s=200, c=IEEE_COLORS['blue'], marker='^', edgecolors='black', linewidth=1)
    ax.text(x, y, z + 5, "📡", fontsize=20, ha='center', va='center', zorder=10)
    ax.text(x, y, z - 5, 'Base Station', ha='center', va='center', fontsize=11, weight='bold')

def draw_mobile_user(ax: Axes3D, pos: tuple):
    """Draws a mobile user with memory-efficient trajectory."""
    x, y, z = pos
    ax.scatter(x, y, z, s=150, c=IEEE_COLORS['red'], marker='o', edgecolors='black', linewidth=1)
    ax.text(x, y, z, "📱", fontsize=20, ha='center', va='center', zorder=10)
    
    # OPTIMIZED: Reduce trajectory points for memory efficiency
    traj_x = np.linspace(x - 100, x, 20)  # Reduced from 50 to 20
    traj_y = np.sin(np.linspace(0, 2*np.pi, 20)) * 5  # Reduced from 50 to 20
    ax.plot(traj_x, traj_y, z, '--', color=IEEE_COLORS['red'], linewidth=1.5)
    ax.text(x, y + 5, z, 'Mobile User', ha='center', va='center', fontsize=11, weight='bold')
    
    # Clean up trajectory arrays
    del traj_x, traj_y

def draw_oam_beam(ax: Axes3D, l: int, start_pos: tuple, end_pos: tuple, num_slices=10):
    """
    Draws a single, visually clear OAM beam using cross-sections.
    OPTIMIZED: Reduced num_slices and added memory cleanup.
    """
    color = OAM_COLORS.get(l, IEEE_COLORS['gray'])
    
    # OPTIMIZED: Reduce number of slices for memory efficiency
    z_points = np.linspace(start_pos[2], end_pos[2], num_slices)
    x_points = np.linspace(start_pos[0], end_pos[0], num_slices)
    y_points = np.linspace(start_pos[1], end_pos[1], num_slices)
    
    ax.plot(x_points, y_points, z_points, '-', color=color, linewidth=0.5, alpha=0.6)

    # OPTIMIZED: Reduce theta resolution for memory efficiency
    theta = np.linspace(0, 2 * np.pi, 25)  # Reduced from 50 to 25
    
    for i in range(num_slices):
        dist_from_bs = x_points[i]
        w_z = get_beam_width(dist_from_bs)
        
        # Calculate beam radius based on OAM mode physics
        if l == 0:
            radius = 0.5 * w_z  # Show central peak for l=0
        else:
            radius = w_z * np.sqrt(abs(l) / 2)  # Peak of donut for l≠0
        
        path_loss_factor = 1 / (1 + (dist_from_bs/150)**2)
        
        # OPTIMIZED: Use more efficient array creation
        ring_x = np.full_like(theta, x_points[i])
        ring_y = radius * np.cos(theta) + y_points[i]
        ring_z = radius * np.sin(theta) + z_points[i]
        
        verts = [list(zip(ring_x, ring_y, ring_z))]
        
        poly = Poly3DCollection(verts, facecolors=color, alpha=0.5 * path_loss_factor, linewidths=0)
        ax.add_collection3d(poly)
    
    # Clean up large arrays
    del z_points, x_points, y_points, theta

def draw_other_mode_paths(ax: Axes3D, start_pos: tuple, end_pos: tuple):
    """Draws simple lines for other available OAM modes to avoid clutter."""
    for mode, color in OAM_COLORS.items():
        if mode not in [1, 3]:
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], 
                    '-', color=color, linewidth=1.0, alpha=0.3, dashes=[5, 5])

def draw_turbulence(ax: Axes3D, path_length: float):
    """
    Draws a 3D scatter plot to represent turbulent eddies.
    OPTIMIZED: Reduced number of eddies and added memory cleanup.
    """
    # OPTIMIZED: Reduce number of eddies for memory efficiency
    num_eddies = 100  # Reduced from 200 to 100
    
    # OPTIMIZED: Use more efficient random generation
    x = np.random.uniform(20, path_length - 20, num_eddies)
    y = np.random.uniform(-15, 15, num_eddies)
    z = np.random.uniform(0, 15, num_eddies)
    
    sizes = (1 + (x / path_length)**2) * 50 
    colors = x / path_length
    
    cmap = plt.get_cmap('viridis')
    ax.scatter(x, y, z, s=sizes, c=colors, cmap=cmap, alpha=0.25, marker='o', edgecolors='none')
    ax.text(path_length / 2, 0, 25, 'Atmospheric Turbulence', ha='center', fontsize=11, weight='bold', color=IEEE_COLORS['cyan'])
    
    # Clean up large arrays
    del x, y, z, sizes, colors

def add_annotations(ax: Axes3D):
    """Adds explanatory text and arrows to the plot."""
    # Handover event text
    ax.text(125, 25, 10, "Handover Triggered\n(SINR drops)", ha='center', va='center', 
            fontsize=10, weight='bold', color=IEEE_COLORS['red'], 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    # 3D arrow using quiver for handover indication
    arrow_start_x, arrow_start_y, arrow_start_z = 125, 20, 8
    arrow_end_x, arrow_end_y, arrow_end_z = 105, 8, 5
    
    arrow_dx = arrow_end_x - arrow_start_x
    arrow_dy = arrow_end_y - arrow_start_y
    arrow_dz = arrow_end_z - arrow_start_z
    
    ax.quiver(arrow_start_x, arrow_start_y, arrow_start_z,
              arrow_dx, arrow_dy, arrow_dz,
              color=IEEE_COLORS['red'], linewidth=1.5, arrow_length_ratio=0.3)

    # Before Handover Label
    ax.text(50, -20, 5, 'Mode l=1 Active', ha='center', color=OAM_COLORS[1], fontsize=10, weight='bold')
    
    # After Handover Label
    ax.text(180, 20, 5, 'Mode l=3 Active', ha='center', color=OAM_COLORS[3], fontsize=10, weight='bold')

    # Key Challenge Label
    ax.text(0, -28, 50, "Challenge: Turbulence\ninduces mode crosstalk", ha='left', va='top', fontsize=9, color=IEEE_COLORS['gray'])

def cleanup_memory():
    """Clean up memory after plotting operations."""
    gc.collect()  # Force garbage collection
    plt.close('all')  # Close all figures

def generate_figure_1():
    """Main function to generate the complete Figure 1 with memory optimization."""
    try:
        setup_ieee_style()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define key positions
        BS_POS = (0, 0, 10)
        USER_POS = (200, 0, 2)

        # Draw components
        draw_base_station(ax, BS_POS)
        draw_mobile_user(ax, USER_POS)
        draw_turbulence(ax, USER_POS[0])
        
        # Visualize primary OAM beams and other paths
        draw_oam_beam(ax, l=1, start_pos=BS_POS, end_pos=USER_POS)
        draw_oam_beam(ax, l=3, start_pos=BS_POS, end_pos=USER_POS)
        draw_other_mode_paths(ax, BS_POS, end_pos=USER_POS)

        # Add text and arrows
        add_annotations(ax)

        # Set view and labels
        ax.view_init(elev=20, azim=-60)
        ax.set_box_aspect((USER_POS[0], 40, 30)) # Control aspect ratio
        
        ax.set_xlabel('Distance (m)', weight='bold')
        ax.set_ylabel('Lateral Position (m)', weight='bold')
        ax.set_zlabel('Height (m)', weight='bold')
        ax.set_title('Figure 1: System Model for OAM-based Communication with Intelligent Handover', weight='bold', pad=20)
        
        # Remove grid planes for a cleaner look
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Save the figure
        output_filename = 'Figure1_SystemModel_MemoryOptimized.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
        
        print(f"✅ Figure 1 generated successfully and saved as '{output_filename}'")
        
    except Exception as e:
        print(f"❌ Error generating figure: {e}")
        raise
    finally:
        # OPTIMIZED: Always cleanup memory
        cleanup_memory()

if __name__ == "__main__":
    generate_figure_1() 