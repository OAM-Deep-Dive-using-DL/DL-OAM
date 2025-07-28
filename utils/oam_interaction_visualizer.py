import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from typing import Optional, Tuple, List, Dict
from utils.oam_visualizer import generate_oam_mode


def generate_oam_superposition(modes: List[int], weights: List[complex], 
                              size: int = 500, beam_width: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a superposition of OAM modes.
    
    Args:
        modes: List of OAM mode numbers
        weights: Complex weights for each mode
        size: Size of the grid (pixels)
        beam_width: Relative beam width
        
    Returns:
        Tuple of (complex field, intensity)
    """
    if len(modes) != len(weights):
        raise ValueError("Number of modes must match number of weights")
    
    # Initialize field
    field = np.zeros((size, size), dtype=complex)
    
    # Add each mode with its weight
    for l, weight in zip(modes, weights):
        mode_field, _ = generate_oam_mode(l, size, beam_width)
        field += weight * mode_field
    
    # Calculate intensity
    intensity = np.abs(field)**2
    
    # Normalize
    intensity = intensity / np.max(intensity)
    
    return field, intensity


def plot_oam_superposition(modes: List[int], weights: List[complex], 
                          size: int = 500, beam_width: float = 0.3,
                          save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Plot the phase and intensity of a superposition of OAM modes.
    
    Args:
        modes: List of OAM mode numbers
        weights: Complex weights for each mode
        size: Size of the grid (pixels)
        beam_width: Relative beam width
        save_path: Path to save the plot, if None, don't save
        show: Whether to display the plot
    """
    field, intensity = generate_oam_superposition(modes, weights, size, beam_width)
    phase = np.angle(field)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot phase
    im0 = axes[0].imshow(phase, cmap='hsv', origin='lower')
    axes[0].set_title(f'OAM Superposition - Phase Pattern')
    axes[0].set_xlabel('x (pixels)')
    axes[0].set_ylabel('y (pixels)')
    plt.colorbar(im0, ax=axes[0], label='Phase (rad)')
    
    # Plot intensity
    im1 = axes[1].imshow(intensity, cmap='viridis', origin='lower')
    axes[1].set_title(f'OAM Superposition - Intensity Pattern')
    axes[1].set_xlabel('x (pixels)')
    axes[1].set_ylabel('y (pixels)')
    plt.colorbar(im1, ax=axes[1], label='Normalized Intensity')
    
    # Add superposition details
    mode_str = " + ".join([f"{w:.2f}×OAM{l}" for l, w in zip(modes, weights)])
    plt.figtext(0.5, 0.01, f"Superposition: {mode_str}", ha='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved OAM superposition visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def apply_turbulence(field: np.ndarray, strength: float = 0.1) -> np.ndarray:
    """
    Apply atmospheric turbulence effects to an OAM field.
    
    Args:
        field: Complex field
        strength: Turbulence strength (0-1)
        
    Returns:
        Distorted complex field
    """
    size = field.shape[0]
    
    # Create random phase screen
    phase_screen = np.random.normal(0, strength, size=(size, size))
    
    # Smooth the phase screen (simple Gaussian filter)
    from scipy.ndimage import gaussian_filter
    phase_screen = gaussian_filter(phase_screen, sigma=size/20)
    
    # Apply phase distortion
    distorted_field = field * np.exp(1j * phase_screen)
    
    return distorted_field


def apply_pointing_error(field: np.ndarray, error_x: float = 0.0, error_y: float = 0.0) -> np.ndarray:
    """
    Apply pointing error to an OAM field.
    
    Args:
        field: Complex field
        error_x: Horizontal pointing error (-1 to 1)
        error_y: Vertical pointing error (-1 to 1)
        
    Returns:
        Shifted complex field
    """
    size = field.shape[0]
    
    # Convert to pixel shifts
    shift_x = int(error_x * size / 4)  # Limit to 1/4 of the image
    shift_y = int(error_y * size / 4)
    
    # Apply shift using numpy roll
    shifted_field = np.roll(field, shift=(shift_y, shift_x), axis=(0, 1))
    
    return shifted_field


def visualize_oam_impairments(l: int, size: int = 500, beam_width: float = 0.3,
                             save_dir: Optional[str] = None, show: bool = True) -> None:
    """
    Visualize the effects of various impairments on an OAM mode.
    
    Args:
        l: OAM mode number
        size: Size of the grid (pixels)
        beam_width: Relative beam width
        save_dir: Directory to save the plots, if None, don't save
        show: Whether to display the plot
    """
    # Generate original OAM mode
    field, intensity = generate_oam_mode(l, size, beam_width)
    
    # Apply different impairments
    impairments = {
        "Original": field,
        "Weak Turbulence": apply_turbulence(field, 0.2),
        "Strong Turbulence": apply_turbulence(field, 0.5),
        "Pointing Error": apply_pointing_error(field, 0.15, 0.15)
    }
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Plot each impairment
    for i, (name, impaired_field) in enumerate(impairments.items()):
        # Calculate intensity
        impaired_intensity = np.abs(impaired_field)**2
        impaired_intensity = impaired_intensity / np.max(impaired_intensity)
        
        # Calculate phase
        impaired_phase = np.angle(impaired_field)
        
        # Plot phase
        im0 = axes[0, i].imshow(impaired_phase, cmap='hsv', origin='lower')
        axes[0, i].set_title(f'{name} - Phase')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Plot intensity
        im1 = axes[1, i].imshow(impaired_intensity, cmap='viridis', origin='lower')
        axes[1, i].set_title(f'{name} - Intensity')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    # Add row labels
    axes[0, 0].set_ylabel('Phase')
    axes[1, 0].set_ylabel('Intensity')
    
    plt.suptitle(f'Effects of Impairments on OAM Mode {l}', fontsize=16)
    plt.tight_layout()
    
    # Save if directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"oam_mode_{l}_impairments.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved OAM impairments visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_oam_mode_coupling(l1: int, l2: int, coupling_strength: float = 0.3,
                               size: int = 500, beam_width: float = 0.3,
                               save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Visualize the coupling between two OAM modes.
    
    Args:
        l1: First OAM mode number
        l2: Second OAM mode number
        coupling_strength: Coupling strength between modes (0-1)
        size: Size of the grid (pixels)
        beam_width: Relative beam width
        save_path: Path to save the plot, if None, don't save
        show: Whether to display the plot
    """
    # Generate individual modes
    field1, intensity1 = generate_oam_mode(l1, size, beam_width)
    field2, intensity2 = generate_oam_mode(l2, size, beam_width)
    
    # Create coupled field
    coupled_field = field1 + coupling_strength * field2
    coupled_intensity = np.abs(coupled_field)**2
    coupled_intensity = coupled_intensity / np.max(coupled_intensity)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot phases
    phase1 = np.angle(field1)
    phase2 = np.angle(field2)
    coupled_phase = np.angle(coupled_field)
    
    im0 = axes[0, 0].imshow(phase1, cmap='hsv', origin='lower')
    axes[0, 0].set_title(f'OAM Mode {l1} - Phase')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    
    im1 = axes[0, 1].imshow(phase2, cmap='hsv', origin='lower')
    axes[0, 1].set_title(f'OAM Mode {l2} - Phase')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    
    im2 = axes[0, 2].imshow(coupled_phase, cmap='hsv', origin='lower')
    axes[0, 2].set_title(f'Coupled Mode - Phase')
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    
    # Plot intensities
    im3 = axes[1, 0].imshow(intensity1, cmap='viridis', origin='lower')
    axes[1, 0].set_title(f'OAM Mode {l1} - Intensity')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    
    im4 = axes[1, 1].imshow(intensity2, cmap='viridis', origin='lower')
    axes[1, 1].set_title(f'OAM Mode {l2} - Intensity')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    im5 = axes[1, 2].imshow(coupled_intensity, cmap='viridis', origin='lower')
    axes[1, 2].set_title(f'Coupled Mode - Intensity')
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    
    # Add colorbars
    plt.colorbar(im0, ax=axes[0, 0], label='Phase (rad)')
    plt.colorbar(im3, ax=axes[1, 0], label='Normalized Intensity')
    
    plt.suptitle(f'Mode Coupling: OAM{l1} + {coupling_strength:.2f}×OAM{l2}', fontsize=16)
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved OAM mode coupling visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_oam_crosstalk_matrix(max_mode: int = 6, distance_factor: float = 0.5,
                                  save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Visualize the crosstalk matrix between OAM modes.
    
    Args:
        max_mode: Maximum OAM mode number
        distance_factor: Distance factor affecting crosstalk (0-1)
        save_path: Path to save the plot, if None, don't save
        show: Whether to display the plot
    """
    # Create crosstalk matrix
    num_modes = max_mode
    crosstalk = np.zeros((num_modes, num_modes))
    
    # Fill crosstalk matrix
    for i in range(num_modes):
        mode_i = i + 1
        for j in range(num_modes):
            if i == j:
                # Main diagonal (self-coupling)
                crosstalk[i, j] = 1.0
            else:
                mode_j = j + 1
                mode_diff = abs(mode_i - mode_j)
                
                # Simple crosstalk model - decreases with mode difference
                if mode_diff == 1:
                    crosstalk[i, j] = 0.1 * distance_factor  # 10% coupling for adjacent modes
                elif mode_diff == 2:
                    crosstalk[i, j] = 0.05 * distance_factor  # 5% coupling for modes 2 steps apart
                else:
                    crosstalk[i, j] = 0.02 * distance_factor / mode_diff  # Lower coupling for distant modes
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot crosstalk matrix as heatmap
    im = ax.imshow(crosstalk, cmap='viridis', origin='lower')
    
    # Add labels
    ax.set_xlabel('Receiving OAM Mode')
    ax.set_ylabel('Transmitting OAM Mode')
    ax.set_title(f'OAM Mode Crosstalk Matrix (Distance Factor: {distance_factor:.1f})')
    
    # Add mode numbers as ticks
    mode_labels = [str(i+1) for i in range(num_modes)]
    ax.set_xticks(np.arange(num_modes))
    ax.set_yticks(np.arange(num_modes))
    ax.set_xticklabels(mode_labels)
    ax.set_yticklabels(mode_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Coupling Strength')
    
    # Add text annotations
    for i in range(num_modes):
        for j in range(num_modes):
            text = ax.text(j, i, f"{crosstalk[i, j]:.2f}",
                          ha="center", va="center", color="w" if crosstalk[i, j] < 0.7 else "black")
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved OAM crosstalk matrix visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Create plots directory
    plots_dir = "plots/oam_modes"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate OAM superposition visualizations
    plot_oam_superposition([1, 3], [1.0, 0.5], save_path=f"{plots_dir}/oam_superposition_1_3.png", show=False)
    plot_oam_superposition([2, 4, 6], [0.7, 0.5, 0.3], save_path=f"{plots_dir}/oam_superposition_2_4_6.png", show=False)
    
    # Generate OAM impairments visualizations
    visualize_oam_impairments(3, save_dir=plots_dir, show=False)
    
    # Generate OAM mode coupling visualizations
    visualize_oam_mode_coupling(1, 2, 0.3, save_path=f"{plots_dir}/oam_coupling_1_2.png", show=False)
    visualize_oam_mode_coupling(3, 4, 0.5, save_path=f"{plots_dir}/oam_coupling_3_4.png", show=False)
    
    # Generate OAM crosstalk matrix visualization
    visualize_oam_crosstalk_matrix(6, 0.5, save_path=f"{plots_dir}/oam_crosstalk_matrix.png", show=False)
    
    print("All OAM interaction visualizations have been generated in the plots/oam_modes directory.") 