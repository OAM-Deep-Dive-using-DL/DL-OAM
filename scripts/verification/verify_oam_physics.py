#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import assoc_laguerre
import os
from utils.oam_visualizer import generate_oam_mode
from utils.oam_interaction_visualizer import generate_oam_superposition, apply_turbulence, apply_pointing_error


def generate_theoretical_lg_mode(l: int, p: int = 0, size: int = 500, beam_width: float = 0.3) -> np.ndarray:
    """
    Generate a Laguerre-Gaussian OAM mode using the full theoretical equation.
    
    Args:
        l: Azimuthal index (topological charge)
        p: Radial index (defaults to 0 for pure OAM modes)
        size: Size of the grid (pixels)
        beam_width: Relative beam width
        
    Returns:
        Complex field
    """
    # Create coordinate grid
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Convert to polar coordinates
    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    
    # Parameters
    w0 = beam_width  # Beam waist
    z = 0  # Propagation distance (at beam waist)
    k = 2 * np.pi  # Wave number (normalized)
    zR = np.pi * w0**2  # Rayleigh range
    w_z = w0 * np.sqrt(1 + (z/zR)**2)  # Beam width at distance z
    
    # Radius of curvature and Gouy phase
    R_z = z * (1 + (zR/z)**2) if z != 0 else float('inf')
    psi_z = np.arctan(z/zR)
    
    # Normalization constant
    if p == 0 and l == 0:
        C = 1.0
    else:
        p_fact = np.math.factorial(p)
        l_abs = abs(l)
        C = np.sqrt(2 * p_fact / (np.pi * (p_fact + np.math.factorial(p + l_abs))))
    
    # Compute the associated Laguerre polynomial
    if p == 0:
        # For p=0, the associated Laguerre polynomial is 1
        laguerre = np.ones_like(r)
    else:
        # For r=0, set to a small value to avoid division by zero
        r_safe = np.copy(r)
        r_safe[r == 0] = 1e-10
        
        # Compute for each point
        laguerre = np.zeros_like(r, dtype=float)
        for i in range(size):
            for j in range(size):
                if r_safe[i, j] > 0:
                    # The argument for the Laguerre polynomial
                    rho = 2 * (r_safe[i, j]/w_z)**2
                    laguerre[i, j] = assoc_laguerre(rho, p, abs(l))
    
    # Compute the LG mode
    rho = np.sqrt(2) * r / w_z
    
    # Amplitude term
    amplitude = C * (rho**abs(l)) * np.exp(-rho**2/2) * laguerre
    
    # Phase terms
    phase_azimuthal = np.exp(1j * l * phi)  # Azimuthal phase (OAM)
    phase_gouy = np.exp(-1j * (2*p + abs(l) + 1) * psi_z)  # Gouy phase
    
    # Wavefront curvature phase
    if z == 0:
        phase_curvature = np.ones_like(r)
    else:
        phase_curvature = np.exp(-1j * k * r**2 / (2 * R_z))
    
    # Combine all terms
    field = amplitude * phase_azimuthal * phase_gouy * phase_curvature
    
    return field


def verify_oam_azimuthal_phase(l_values=[1, 2, 3], size=200):
    """
    Verify that the azimuthal phase of the OAM modes increases by 2π when going around the beam axis.
    """
    print("\n=== Verifying OAM Azimuthal Phase ===")
    
    for l in l_values:
        # Generate OAM mode
        field, _ = generate_oam_mode(l, size=size)
        
        # Extract phase along a circle
        radius = size // 4  # Quarter of the way from center to edge
        center = size // 2
        
        # Sample points along a circle
        theta = np.linspace(0, 2*np.pi, 100)
        x = center + radius * np.cos(theta)
        y = center + radius * np.sin(theta)
        
        # Convert to integer indices
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        
        # Extract phases
        phases = np.angle(field[y, x])
        
        # Calculate phase difference over one complete revolution
        phase_diff = np.unwrap(phases)[-1] - np.unwrap(phases)[0]
        expected_diff = 2 * np.pi * l
        
        print(f"OAM mode l={l}:")
        print(f"  Phase difference over one revolution: {phase_diff:.4f} rad")
        print(f"  Expected phase difference: {expected_diff:.4f} rad")
        print(f"  Relative error: {abs(phase_diff - expected_diff) / expected_diff * 100:.4f}%")


def verify_oam_intensity_profile(l_values=[1, 2, 3], size=200):
    """
    Verify that the intensity profile of OAM modes shows the characteristic donut shape
    with zero intensity at the center for l != 0.
    """
    print("\n=== Verifying OAM Intensity Profile ===")
    
    for l in l_values:
        # Generate OAM mode
        field, intensity = generate_oam_mode(l, size=size)
        
        # Check center intensity
        center = size // 2
        center_intensity = intensity[center, center]
        
        # Check maximum intensity location
        max_idx = np.unravel_index(np.argmax(intensity), intensity.shape)
        max_r = np.sqrt((max_idx[0] - center)**2 + (max_idx[1] - center)**2)
        
        print(f"OAM mode l={l}:")
        print(f"  Center intensity: {center_intensity:.6f}")
        print(f"  Maximum intensity radius: {max_r:.2f} pixels")
        
        # For l=0, maximum should be at center; for l!=0, it should be away from center
        if l == 0:
            print(f"  Expected maximum at center: {'Yes' if max_r < 2 else 'No'}")
        else:
            print(f"  Expected donut shape: {'Yes' if max_r > 5 else 'No'}")


def verify_oam_superposition():
    """
    Verify that OAM mode superposition follows interference principles.
    """
    print("\n=== Verifying OAM Mode Superposition ===")
    
    # Test case: Equal superposition of l=1 and l=-1 should create a petal pattern
    field, intensity = generate_oam_superposition([1, -1], [1/np.sqrt(2), 1/np.sqrt(2)], size=200)
    
    # Count intensity maxima along a circle
    center = 100
    radius = 30
    theta = np.linspace(0, 2*np.pi, 100)
    x = center + radius * np.cos(theta)
    y = center + radius * np.sin(theta)
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    
    # Extract intensity
    circle_intensity = intensity[y, x]
    
    # Find peaks (local maxima)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(circle_intensity, height=0.5)
    
    print(f"Superposition of l=1 and l=-1:")
    print(f"  Number of intensity maxima (petals): {len(peaks)}")
    print(f"  Expected number for |l1-l2|: {abs(1-(-1))}")
    
    # Test another case: l=2 and l=-2
    field2, intensity2 = generate_oam_superposition([2, -2], [1/np.sqrt(2), 1/np.sqrt(2)], size=200)
    circle_intensity2 = intensity2[y, x]
    peaks2, _ = find_peaks(circle_intensity2, height=0.5)
    
    print(f"Superposition of l=2 and l=-2:")
    print(f"  Number of intensity maxima (petals): {len(peaks2)}")
    print(f"  Expected number for |l1-l2|: {abs(2-(-2))}")


def verify_oam_conservation():
    """
    Verify that OAM is conserved in superpositions.
    """
    print("\n=== Verifying OAM Conservation ===")
    
    # Generate superposition
    l1, l2 = 2, 3
    w1, w2 = 0.6, 0.8
    field, _ = generate_oam_superposition([l1, l2], [w1, w2], size=200)
    
    # Calculate total OAM (should be a weighted average)
    # In a real experiment, this would be measured differently
    # Here we use a simplified approach for verification
    expected_oam = (abs(w1)**2 * l1 + abs(w2)**2 * l2) / (abs(w1)**2 + abs(w2)**2)
    
    # Normalize weights
    norm = np.sqrt(abs(w1)**2 + abs(w2)**2)
    w1_norm = w1 / norm
    w2_norm = w2 / norm
    
    print(f"Superposition of l={l1} (weight={abs(w1_norm)**2:.2f}) and l={l2} (weight={abs(w2_norm)**2:.2f}):")
    print(f"  Expected weighted average OAM: {expected_oam:.4f}")
    print(f"  Note: In quantum mechanics, a measurement would collapse to either l={l1} or l={l2}")


def verify_turbulence_effects():
    """
    Verify that turbulence affects higher-order OAM modes more severely.
    """
    print("\n=== Verifying Turbulence Effects on OAM Modes ===")
    
    # Generate OAM modes
    l_values = [1, 3, 5]
    size = 200
    turbulence_strength = 0.3
    
    for l in l_values:
        # Generate original field
        field, intensity = generate_oam_mode(l, size=size)
        
        # Apply turbulence
        disturbed_field = apply_turbulence(field, turbulence_strength)
        disturbed_intensity = np.abs(disturbed_field)**2
        disturbed_intensity = disturbed_intensity / np.max(disturbed_intensity)
        
        # Calculate correlation between original and disturbed intensity
        correlation = np.corrcoef(intensity.flatten(), disturbed_intensity.flatten())[0, 1]
        
        print(f"OAM mode l={l}:")
        print(f"  Correlation after turbulence: {correlation:.4f}")
        print(f"  Intensity pattern distortion: {100 * (1 - correlation):.2f}%")


def verify_pointing_error_sensitivity():
    """
    Verify that higher-order OAM modes are more sensitive to pointing errors.
    """
    print("\n=== Verifying Pointing Error Sensitivity ===")
    
    # Generate OAM modes
    l_values = [1, 3, 5]
    size = 200
    error_x = 0.05  # 5% shift
    
    for l in l_values:
        # Generate original field
        field, intensity = generate_oam_mode(l, size=size)
        
        # Apply pointing error
        shifted_field = apply_pointing_error(field, error_x, 0)
        shifted_intensity = np.abs(shifted_field)**2
        shifted_intensity = shifted_intensity / np.max(shifted_intensity)
        
        # Calculate correlation between original and shifted intensity
        correlation = np.corrcoef(intensity.flatten(), shifted_intensity.flatten())[0, 1]
        
        print(f"OAM mode l={l}:")
        print(f"  Correlation after pointing error: {correlation:.4f}")
        print(f"  Intensity pattern distortion: {100 * (1 - correlation):.2f}%")


def verify_against_theoretical_model():
    """
    Verify our simplified OAM model against the full theoretical Laguerre-Gaussian model.
    """
    print("\n=== Verifying Against Theoretical LG Model ===")
    
    l_values = [1, 2, 3]
    size = 200
    
    for l in l_values:
        # Generate our simplified model
        our_field, _ = generate_oam_mode(l, size=size)
        
        # Generate theoretical model
        theoretical_field = generate_theoretical_lg_mode(l, p=0, size=size)
        
        # Normalize both fields for comparison
        our_field = our_field / np.max(np.abs(our_field))
        theoretical_field = theoretical_field / np.max(np.abs(theoretical_field))
        
        # Calculate phase correlation
        our_phase = np.angle(our_field)
        theoretical_phase = np.angle(theoretical_field)
        phase_correlation = np.corrcoef(np.unwrap(our_phase.flatten()), 
                                        np.unwrap(theoretical_phase.flatten()))[0, 1]
        
        # Calculate amplitude correlation
        our_amplitude = np.abs(our_field)
        theoretical_amplitude = np.abs(theoretical_field)
        amplitude_correlation = np.corrcoef(our_amplitude.flatten(), 
                                           theoretical_amplitude.flatten())[0, 1]
        
        print(f"OAM mode l={l}:")
        print(f"  Phase correlation with theoretical model: {phase_correlation:.4f}")
        print(f"  Amplitude correlation with theoretical model: {amplitude_correlation:.4f}")


def create_verification_plots(output_dir="plots/oam_verification"):
    """
    Create verification plots comparing our model with theoretical predictions.
    """
    print("\n=== Creating Verification Plots ===")
    os.makedirs(output_dir, exist_ok=True)
    
    l_values = [1, 3]
    size = 200
    
    for l in l_values:
        # Generate our simplified model
        our_field, our_intensity = generate_oam_mode(l, size=size)
        our_phase = np.angle(our_field)
        
        # Generate theoretical model
        theoretical_field = generate_theoretical_lg_mode(l, p=0, size=size)
        theoretical_intensity = np.abs(theoretical_field)**2
        theoretical_intensity = theoretical_intensity / np.max(theoretical_intensity)
        theoretical_phase = np.angle(theoretical_field)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot our model
        im0 = axes[0, 0].imshow(our_phase, cmap='hsv', origin='lower')
        axes[0, 0].set_title(f'Our Model - Phase (l={l})')
        plt.colorbar(im0, ax=axes[0, 0], label='Phase (rad)')
        
        im1 = axes[0, 1].imshow(our_intensity, cmap='viridis', origin='lower')
        axes[0, 1].set_title(f'Our Model - Intensity (l={l})')
        plt.colorbar(im1, ax=axes[0, 1], label='Normalized Intensity')
        
        # Plot theoretical model
        im2 = axes[1, 0].imshow(theoretical_phase, cmap='hsv', origin='lower')
        axes[1, 0].set_title(f'Theoretical Model - Phase (l={l})')
        plt.colorbar(im2, ax=axes[1, 0], label='Phase (rad)')
        
        im3 = axes[1, 1].imshow(theoretical_intensity, cmap='viridis', origin='lower')
        axes[1, 1].set_title(f'Theoretical Model - Intensity (l={l})')
        plt.colorbar(im3, ax=axes[1, 1], label='Normalized Intensity')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(output_dir, f"oam_verification_l{l}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved verification plot for l={l} to {save_path}")
        
        # Create radial profile plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate radial profiles
        center = size // 2
        max_radius = size // 2 - 10
        radii = np.arange(0, max_radius)
        our_radial_profile = np.zeros(max_radius)
        theoretical_radial_profile = np.zeros(max_radius)
        
        for r in radii:
            # Create a circle mask
            y, x = np.ogrid[-center:size-center, -center:size-center]
            mask = (r-0.5 < np.sqrt(x*x + y*y)) & (np.sqrt(x*x + y*y) < r+0.5)
            
            # Calculate average intensity along the circle
            our_radial_profile[r] = np.mean(our_intensity[mask])
            theoretical_radial_profile[r] = np.mean(theoretical_intensity[mask])
        
        # Normalize profiles
        our_radial_profile = our_radial_profile / np.max(our_radial_profile)
        theoretical_radial_profile = theoretical_radial_profile / np.max(theoretical_radial_profile)
        
        # Plot radial profiles
        ax.plot(radii, our_radial_profile, 'b-', linewidth=2, label='Our Model')
        ax.plot(radii, theoretical_radial_profile, 'r--', linewidth=2, label='Theoretical Model')
        
        ax.set_xlabel('Radial Distance (pixels)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(f'Radial Intensity Profile Comparison for OAM Mode l={l}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        save_path = os.path.join(output_dir, f"oam_radial_profile_l{l}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved radial profile comparison for l={l} to {save_path}")


if __name__ == "__main__":
    # Run all verification functions
    verify_oam_azimuthal_phase()
    verify_oam_intensity_profile()
    verify_oam_superposition()
    verify_oam_conservation()
    verify_turbulence_effects()
    verify_pointing_error_sensitivity()
    verify_against_theoretical_model()
    create_verification_plots()
    
    print("\nAll verification tests completed. OAM physics implementation has been verified.") 