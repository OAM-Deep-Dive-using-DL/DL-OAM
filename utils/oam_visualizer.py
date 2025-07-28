import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from typing import Optional, Tuple, List


def generate_oam_mode(l: int, size: int = 500, beam_width: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Laguerre-Gaussian OAM mode with topological charge l.
    
    Args:
        l: Topological charge (OAM mode number)
        size: Size of the grid (pixels)
        beam_width: Relative beam width
        
    Returns:
        Tuple of (complex field, intensity)
    """
    # Create coordinate grid
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Convert to polar coordinates
    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    
    # Generate LG beam (simplified model)
    # Using only the azimuthal phase term and Gaussian amplitude
    w = beam_width  # Beam waist
    amplitude = np.exp(-r**2 / w**2) * (r/w)**(abs(l))
    phase = np.exp(1j * l * phi)
    
    # Create complex field
    field = amplitude * phase
    
    # Calculate intensity
    intensity = np.abs(field)**2
    
    # Normalize
    intensity = intensity / np.max(intensity)
    
    return field, intensity


def plot_oam_mode(l: int, size: int = 500, beam_width: float = 0.3, 
                  save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Plot the phase and intensity of an OAM mode.
    
    Args:
        l: Topological charge (OAM mode number)
        size: Size of the grid (pixels)
        beam_width: Relative beam width
        save_path: Path to save the plot, if None, don't save
        show: Whether to display the plot
    """
    field, intensity = generate_oam_mode(l, size, beam_width)
    phase = np.angle(field)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot phase
    im0 = axes[0].imshow(phase, cmap='hsv', origin='lower')
    axes[0].set_title(f'OAM Mode {l} - Phase Pattern')
    axes[0].set_xlabel('x (pixels)')
    axes[0].set_ylabel('y (pixels)')
    plt.colorbar(im0, ax=axes[0], label='Phase (rad)')
    
    # Plot intensity
    im1 = axes[1].imshow(intensity, cmap='viridis', origin='lower')
    axes[1].set_title(f'OAM Mode {l} - Intensity Pattern')
    axes[1].set_xlabel('x (pixels)')
    axes[1].set_ylabel('y (pixels)')
    plt.colorbar(im1, ax=axes[1], label='Normalized Intensity')
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved OAM mode {l} visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_oam_modes(modes: List[int], size: int = 500, beam_width: float = 0.3,
                           save_dir: Optional[str] = None, show: bool = True) -> None:
    """
    Plot multiple OAM modes in a grid.
    
    Args:
        modes: List of OAM mode numbers to plot
        size: Size of the grid (pixels)
        beam_width: Relative beam width
        save_dir: Directory to save the plots, if None, don't save
        show: Whether to display the plot
    """
    n_modes = len(modes)
    fig, axes = plt.subplots(2, n_modes, figsize=(4*n_modes, 8))
    
    for i, l in enumerate(modes):
        field, intensity = generate_oam_mode(l, size, beam_width)
        phase = np.angle(field)
        
        # Plot phase
        im0 = axes[0, i].imshow(phase, cmap='hsv', origin='lower')
        axes[0, i].set_title(f'OAM Mode {l} - Phase')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Plot intensity
        im1 = axes[1, i].imshow(intensity, cmap='viridis', origin='lower')
        axes[1, i].set_title(f'OAM Mode {l} - Intensity')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    # Add colorbars
    cbar_ax0 = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    cbar_ax1 = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    fig.colorbar(im0, cax=cbar_ax0, label='Phase (rad)')
    fig.colorbar(im1, cax=cbar_ax1, label='Normalized Intensity')
    
    plt.suptitle('OAM Mode Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save if directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"oam_modes_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved OAM modes comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_oam_mode_3d(l: int, size: int = 200, beam_width: float = 0.3,
                    save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Create a 3D visualization of an OAM mode intensity pattern.
    
    Args:
        l: Topological charge (OAM mode number)
        size: Size of the grid (pixels)
        beam_width: Relative beam width
        save_path: Path to save the plot, if None, don't save
        show: Whether to display the plot
    """
    # Generate the OAM mode
    field, intensity = generate_oam_mode(l, size, beam_width)
    
    # Create coordinate grid
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, intensity, cmap=cm.plasma, linewidth=0, antialiased=True)
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    ax.set_title(f'OAM Mode {l} - 3D Intensity Pattern')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Normalized Intensity')
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved OAM mode {l} 3D visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_oam_propagation(l: int, distances: List[float], size: int = 200, beam_width: float = 0.3,
                             save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Visualize how an OAM mode propagates and evolves over distance.
    
    Args:
        l: Topological charge (OAM mode number)
        distances: List of propagation distances (normalized)
        size: Size of the grid (pixels)
        beam_width: Initial beam width
        save_path: Path to save the plot, if None, don't save
        show: Whether to display the plot
    """
    n_distances = len(distances)
    fig, axes = plt.subplots(2, n_distances, figsize=(4*n_distances, 8))
    
    for i, z in enumerate(distances):
        # Beam width increases with distance
        w_z = beam_width * np.sqrt(1 + (z/np.pi/beam_width**2)**2)
        
        # Generate the OAM mode at this distance
        field, intensity = generate_oam_mode(l, size, w_z)
        phase = np.angle(field)
        
        # Plot phase
        im0 = axes[0, i].imshow(phase, cmap='hsv', origin='lower')
        axes[0, i].set_title(f'Distance = {z:.1f}')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Plot intensity
        im1 = axes[1, i].imshow(intensity, cmap='viridis', origin='lower')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    # Add row labels
    axes[0, 0].set_ylabel('Phase')
    axes[1, 0].set_ylabel('Intensity')
    
    # Add colorbars
    cbar_ax0 = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    cbar_ax1 = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    fig.colorbar(im0, cax=cbar_ax0, label='Phase (rad)')
    fig.colorbar(im1, cax=cbar_ax1, label='Normalized Intensity')
    
    plt.suptitle(f'OAM Mode {l} Propagation', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved OAM mode {l} propagation to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Create plots directory
    plots_dir = "plots/oam_modes"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate individual mode plots
    for l in range(1, 7):
        plot_oam_mode(l, save_path=f"{plots_dir}/oam_mode_{l}.png", show=False)
    
    # Generate comparison of multiple modes
    plot_multiple_oam_modes([1, 2, 3, 4, 5, 6], save_dir=plots_dir, show=False)
    
    # Generate 3D visualizations
    for l in [1, 3, 5]:
        plot_oam_mode_3d(l, save_path=f"{plots_dir}/oam_mode_{l}_3d.png", show=False)
    
    # Generate propagation visualization
    visualize_oam_propagation(3, [0, 1, 2, 3, 4], save_path=f"{plots_dir}/oam_mode_3_propagation.png", show=False)
    
    print("All OAM mode visualizations have been generated in the plots/oam_modes directory.") 