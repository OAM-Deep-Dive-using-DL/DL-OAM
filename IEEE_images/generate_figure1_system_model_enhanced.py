#!/usr/bin/env python3
"""
IEEE Format Figure 1: System Model & Problem Definition (ENHANCED MEMORY MANAGEMENT)
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
- ENHANCED MEMORY MANAGEMENT:
  - Aggressive garbage collection
  - Memory monitoring
  - Optimized array creation and deletion
  - Reduced resolution where visually acceptable
  - Context manager for automatic cleanup

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
import os
import psutil  # For memory monitoring
import contextlib
import time
from typing import Tuple, List, Optional, Dict, Any
from scipy.special import genlaguerre, factorial
import math
import warnings

# Suppress matplotlib warnings about font substitution
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# --- Memory Management Utilities ---

class MemoryMonitor:
    """Monitor and report memory usage during figure generation."""
    
    def __init__(self, enabled: bool = True):
        """Initialize the memory monitor.
        
        Args:
            enabled: Whether to enable memory monitoring
        """
        self.enabled = enabled
        self.process = psutil.Process(os.getpid())
        self.start_memory = self.get_memory_mb() if enabled else 0
        self.peak_memory = self.start_memory
        self.checkpoints = {}
    
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if not self.enabled:
            return 0.0
        return self.process.memory_info().rss / (1024 * 1024)
    
    def checkpoint(self, name: str):
        """Record memory usage at a checkpoint."""
        if not self.enabled:
            return
        
        current = self.get_memory_mb()
        self.checkpoints[name] = current
        self.peak_memory = max(self.peak_memory, current)
        
        # Print memory usage
        print(f"Memory at {name}: {current:.2f} MB (Δ: {current - self.start_memory:.2f} MB)")
    
    def report(self):
        """Print a final memory usage report."""
        if not self.enabled:
            return
            
        current = self.get_memory_mb()
        print("\n=== Memory Usage Report ===")
        print(f"Starting memory: {self.start_memory:.2f} MB")
        print(f"Peak memory: {self.peak_memory:.2f} MB")
        print(f"Final memory: {current:.2f} MB")
        print(f"Memory increase: {current - self.start_memory:.2f} MB")
        print(f"Memory recovered from peak: {self.peak_memory - current:.2f} MB")
        print("=========================\n")


@contextlib.contextmanager
def figure_context(figsize: Tuple[float, float] = (12, 8), memory_monitor: Optional[MemoryMonitor] = None):
    """Context manager for creating and cleaning up matplotlib figures.
    
    Args:
        figsize: Figure size (width, height) in inches
        memory_monitor: Optional memory monitor for tracking usage
    
    Yields:
        Tuple of (figure, axis)
    """
    if memory_monitor:
        memory_monitor.checkpoint("Before figure creation")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if memory_monitor:
        memory_monitor.checkpoint("After figure creation")
    
    try:
        yield fig, ax
    finally:
        # Clean up
        if memory_monitor:
            memory_monitor.checkpoint("Before cleanup")
        
        plt.close(fig)
        gc.collect()
        
        if memory_monitor:
            memory_monitor.checkpoint("After cleanup")


def cleanup_arrays(*arrays):
    """Delete arrays and collect garbage.
    
    Args:
        *arrays: Arrays to delete
    """
    for arr in arrays:
        del arr
    gc.collect()

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
    # Handle l=0 case (fundamental Gaussian)
    if l == 0 and p == 0:
        # E_00 = (w0/w(z)) * exp(-r²/w²(z))
        # For p=0, l=0, the normalization factor C = sqrt(2/π)
        normalization = math.sqrt(2/math.pi)
        amplitude = normalization * np.exp(-r**2 / w_z**2)
        return np.abs(amplitude)**2
    
    # For higher order modes (l≠0 or p≠0)
    # Calculate normalization factor C = sqrt(2*p!/(π(p+|l|)!))
    # Use lgamma for large factorials to avoid overflow
    if p > 10 or abs(l) > 10:
        log_normalization = 0.5 * (math.log(2) + math.lgamma(p+1) - math.log(math.pi) - math.lgamma(p + abs(l) + 1))
        normalization = math.exp(log_normalization)
    else:
        normalization = math.sqrt(2 * factorial(p) / (math.pi * factorial(p + abs(l))))
    
    # Calculate the normalized LG beam amplitude
    rho = r / w_z  # Normalized radial coordinate
    amplitude = normalization * (math.sqrt(2) * rho)**abs(l) * np.exp(-rho**2)
    
    # Apply the generalized Laguerre polynomial if p > 0
    if p > 0:
        laguerre = genlaguerre(p, abs(l))(2 * rho**2)
        amplitude = amplitude * laguerre
    
    # Return intensity (squared amplitude)
    return np.abs(amplitude)**2

def get_beam_width(z: float, w0: float = 0.05, wavelength: float = 1.07e-2) -> float:
    """
    Calculate the beam width at distance z using the proper beam propagation formula.
    
    Args:
        z: Distance from beam waist (m)
        w0: Beam waist (minimum beam width at focus) (m)
        wavelength: Wavelength of the beam (m)
        
    Returns:
        Beam width at distance z (m)
    """
    # Rayleigh range
    z_R = math.pi * w0**2 / wavelength
    
    # Beam width evolution formula
    w_z = w0 * math.sqrt(1 + (z / z_R)**2)
    
    return w_z

# --- Drawing Functions ---

def draw_base_station(ax: Axes3D, pos: tuple, memory_monitor: Optional[MemoryMonitor] = None):
    """Draw the base station."""
    if memory_monitor:
        memory_monitor.checkpoint("Before draw_base_station")
    
    # Draw a simple tower
    height = 15
    width = 3
    ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [0, height], 'k-', linewidth=2)
    ax.plot([pos[0], pos[0]], [pos[1]-width/2, pos[1]+width/2], [height, height], 'k-', linewidth=2)
    ax.text(pos[0], pos[1], height+2, 'Base Station', ha='center', fontsize=10, weight='bold')
    
    if memory_monitor:
        memory_monitor.checkpoint("After draw_base_station")

def draw_mobile_user(ax: Axes3D, pos: tuple, memory_monitor: Optional[MemoryMonitor] = None):
    """Draw the mobile user."""
    if memory_monitor:
        memory_monitor.checkpoint("Before draw_mobile_user")
    
    # Draw a simple user device
    ax.scatter([pos[0]], [pos[1]], [pos[2]], color='k', s=50)
    ax.text(pos[0], pos[1], pos[2]+3, 'Mobile User', ha='center', fontsize=10, weight='bold')
    
    if memory_monitor:
        memory_monitor.checkpoint("After draw_mobile_user")

def draw_oam_beam(ax: Axes3D, l: int, start_pos: tuple, end_pos: tuple, num_slices: int = 8, 
                 memory_monitor: Optional[MemoryMonitor] = None):
    """
    Draws a single, visually clear OAM beam using cross-sections.
    ENHANCED: Further reduced num_slices and added aggressive memory cleanup.
    
    Args:
        ax: 3D axis to draw on
        l: OAM mode number
        start_pos: Starting position (x, y, z)
        end_pos: Ending position (x, y, z)
        num_slices: Number of beam slices to draw
        memory_monitor: Optional memory monitor
    """
    if memory_monitor:
        memory_monitor.checkpoint(f"Before draw_oam_beam l={l}")
    
    color = OAM_COLORS.get(l, IEEE_COLORS['gray'])
    
    # ENHANCED: Create arrays only when needed and with minimal size
    z_points = np.linspace(start_pos[2], end_pos[2], num_slices)
    x_points = np.linspace(start_pos[0], end_pos[0], num_slices)
    y_points = np.linspace(start_pos[1], end_pos[1], num_slices)
    
    # Draw central beam axis
    ax.plot(x_points, y_points, z_points, '-', color=color, linewidth=0.5, alpha=0.6)
    
    # ENHANCED: Reduced theta resolution further
    theta = np.linspace(0, 2 * np.pi, 20)  # Reduced from 25 to 20
    
    # Draw beam slices one at a time with memory cleanup between slices
    for i in range(num_slices):
        dist_from_bs = x_points[i]
        w_z = get_beam_width(dist_from_bs)
        
        # Calculate beam radius based on OAM mode physics
        if l == 0:
            radius = 0.5 * w_z  # Show central peak for l=0
        else:
            radius = w_z * np.sqrt(abs(l) / 2)  # Peak of donut for l≠0
        
        path_loss_factor = 1 / (1 + (dist_from_bs/150)**2)
        
        # ENHANCED: Create ring coordinates efficiently
        ring_x = np.full_like(theta, x_points[i])
        ring_y = radius * np.cos(theta) + y_points[i]
        ring_z = radius * np.sin(theta) + z_points[i]
        
        # Create and add polygon
        verts = [list(zip(ring_x, ring_y, ring_z))]
        poly = Poly3DCollection(verts, facecolors=color, alpha=0.5 * path_loss_factor, linewidths=0)
        ax.add_collection3d(poly)
        
        # ENHANCED: Clean up temporary arrays after each slice
        cleanup_arrays(ring_x, ring_y, ring_z, verts)
    
    # Clean up remaining arrays
    cleanup_arrays(z_points, x_points, y_points, theta)
    
    if memory_monitor:
        memory_monitor.checkpoint(f"After draw_oam_beam l={l}")

def draw_other_mode_paths(ax: Axes3D, start_pos: tuple, end_pos: tuple, 
                         memory_monitor: Optional[MemoryMonitor] = None):
    """Draws simple lines for other available OAM modes to avoid clutter."""
    if memory_monitor:
        memory_monitor.checkpoint("Before draw_other_mode_paths")
    
    for mode, color in OAM_COLORS.items():
        if mode not in [1, 3]:
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], 
                    '-', color=color, linewidth=1.0, alpha=0.3, dashes=[5, 5])
    
    if memory_monitor:
        memory_monitor.checkpoint("After draw_other_mode_paths")

def draw_turbulence(ax: Axes3D, path_length: float, memory_monitor: Optional[MemoryMonitor] = None):
    """
    Draws a 3D scatter plot to represent turbulent eddies.
    ENHANCED: Further reduced number of eddies and implemented batch processing.
    
    Args:
        ax: 3D axis to draw on
        path_length: Length of the path in meters
        memory_monitor: Optional memory monitor
    """
    if memory_monitor:
        memory_monitor.checkpoint("Before draw_turbulence")
    
    # ENHANCED: Further reduce number of eddies for memory efficiency
    num_eddies = 75  # Reduced from 100 to 75
    
    # ENHANCED: Process eddies in smaller batches to reduce peak memory usage
    batch_size = 25
    num_batches = (num_eddies + batch_size - 1) // batch_size  # Ceiling division
    
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_eddies)
        batch_count = end_idx - start_idx
        
        # Generate batch of eddies
        x = np.random.uniform(20, path_length - 20, batch_count)
        y = np.random.uniform(-15, 15, batch_count)
        z = np.random.uniform(0, 15, batch_count)
        
        sizes = (1 + (x / path_length)**2) * 50 
        colors = x / path_length
        
        # Draw this batch
        ax.scatter(x, y, z, s=sizes, c=colors, cmap='viridis', alpha=0.25, marker='o', edgecolors='none')
        
        # Clean up batch arrays immediately
        cleanup_arrays(x, y, z, sizes, colors)
    
    # Add label
    ax.text(path_length / 2, 0, 25, 'Atmospheric Turbulence', ha='center', fontsize=11, weight='bold', color=IEEE_COLORS['cyan'])
    
    if memory_monitor:
        memory_monitor.checkpoint("After draw_turbulence")

def add_annotations(ax: Axes3D, memory_monitor: Optional[MemoryMonitor] = None):
    """Adds explanatory text and arrows to the plot."""
    if memory_monitor:
        memory_monitor.checkpoint("Before add_annotations")
    
    # Handover event text
    ax.text(125, 25, 10, "Handover Triggered\n(SINR drops)", ha='center', va='center', 
            fontsize=10, weight='bold', color=IEEE_COLORS['red'], 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    # 3D arrow using quiver for handover indication
    arrow_start_x, arrow_start_y, arrow_start_z = 125, 20, 8
    arrow_dx, arrow_dy, arrow_dz = -20, -12, -3
    
    ax.quiver(arrow_start_x, arrow_start_y, arrow_start_z,
              arrow_dx, arrow_dy, arrow_dz,
              color=IEEE_COLORS['red'], linewidth=1.5, arrow_length_ratio=0.3)

    # Before Handover Label
    ax.text(50, -20, 5, 'Mode l=1 Active', ha='center', color=OAM_COLORS[1], fontsize=10, weight='bold')
    
    # After Handover Label
    ax.text(180, 20, 5, 'Mode l=3 Active', ha='center', color=OAM_COLORS[3], fontsize=10, weight='bold')

    # Key Challenge Label
    ax.text(0, -28, 50, "Challenge: Turbulence\ninduces mode crosstalk", ha='left', va='top', fontsize=9, color=IEEE_COLORS['gray'])
    
    if memory_monitor:
        memory_monitor.checkpoint("After add_annotations")

def generate_figure_1(output_filename: str = 'Figure1_SystemModel_EnhancedMemory.png', 
                     enable_memory_monitoring: bool = True):
    """
    Main function to generate the complete Figure 1 with enhanced memory management.
    
    Args:
        output_filename: Name of the output file
        enable_memory_monitoring: Whether to enable memory usage monitoring
    """
    # Initialize memory monitor
    memory_monitor = MemoryMonitor(enabled=enable_memory_monitoring)
    memory_monitor.checkpoint("Start")
    
    # Setup IEEE style
    setup_ieee_style()
    memory_monitor.checkpoint("After style setup")
    
    # Define key positions
    BS_POS = (0, 0, 10)
    USER_POS = (200, 0, 2)
    
    # Use context manager for figure creation and cleanup
    with figure_context(figsize=(12, 8), memory_monitor=memory_monitor) as (fig, ax):
        # Draw components with memory monitoring
        draw_base_station(ax, BS_POS, memory_monitor)
        draw_mobile_user(ax, USER_POS, memory_monitor)
        
        # Force garbage collection between major drawing operations
        gc.collect()
        memory_monitor.checkpoint("After base station and user")
        
        # Draw turbulence
        draw_turbulence(ax, USER_POS[0], memory_monitor)
        gc.collect()
        memory_monitor.checkpoint("After turbulence")
        
        # Draw primary OAM beams
        draw_oam_beam(ax, l=1, start_pos=BS_POS, end_pos=USER_POS, memory_monitor=memory_monitor)
        gc.collect()
        memory_monitor.checkpoint("After beam 1")
        
        draw_oam_beam(ax, l=3, start_pos=BS_POS, end_pos=USER_POS, memory_monitor=memory_monitor)
        gc.collect()
        memory_monitor.checkpoint("After beam 3")
        
        # Draw other paths
        draw_other_mode_paths(ax, BS_POS, end_pos=USER_POS, memory_monitor=memory_monitor)
        gc.collect()
        memory_monitor.checkpoint("After other paths")
        
        # Add annotations
        add_annotations(ax, memory_monitor)
        gc.collect()
        memory_monitor.checkpoint("After annotations")
        
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
        
        memory_monitor.checkpoint("Before saving")
        
        # Save the figure
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
        memory_monitor.checkpoint("After saving")
        
        print(f"✅ Figure 1 generated successfully and saved as '{output_filename}'")
    
    # Final cleanup
    plt.close('all')
    gc.collect()
    memory_monitor.checkpoint("Final")
    
    # Print memory report
    memory_monitor.report()

if __name__ == "__main__":
    generate_figure_1()