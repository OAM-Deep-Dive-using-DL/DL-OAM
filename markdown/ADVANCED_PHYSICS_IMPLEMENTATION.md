# Advanced Physics Implementation for OAM Channel Simulator

This document explains the implementation of advanced physics features in the OAM channel simulator for more accurate atmospheric turbulence modeling.

## Overview

The advanced physics implementation includes:

1. FFT-based phase screen generation (McGlamery method)
2. Non-Kolmogorov turbulence support
3. Multi-layer atmospheric modeling
4. Enhanced aperture averaging
5. Inner/outer scale turbulence effects

## 1. FFT-Based Phase Screen Generation (McGlamery Method)

The FFT-based phase screen generation method creates more realistic atmospheric turbulence effects compared to the simpler direct method. The implementation follows these steps:

1. Create a 2D grid of spatial frequencies
2. Generate a power spectral density (PSD) function based on Kolmogorov or non-Kolmogorov spectrum
3. Apply inner/outer scale modifications to the PSD
4. Generate a complex Gaussian random field with the PSD as the variance
5. Perform an inverse FFT to get the phase screen

```python
def _generate_fft_phase_screen(self, r0: float, distance: float) -> np.ndarray:
    # Set up grid parameters
    N = self.phase_screen_resolution  # Grid size
    L = self.phase_screen_size  # Physical size (meters)
    
    # Create coordinate grids and spatial frequencies
    # ...
    
    # Generate power spectrum
    psd = self._kolmogorov_psd(f, r0)  # or _non_kolmogorov_psd
    
    # Apply inner/outer scale modifications
    psd = self._apply_scale_limits(f, psd)
    
    # Generate complex Gaussian random field
    # ...
    
    # Perform inverse FFT to get phase screen
    phase_screen = np.real(np.fft.ifft2(np.fft.ifftshift(complex_field)))
    
    return phase_screen
```

## 2. Non-Kolmogorov Turbulence Support

The standard Kolmogorov spectrum assumes a spectral index of 11/3, but real atmospheric turbulence can deviate from this value. The non-Kolmogorov implementation allows for a configurable spectral index:

```python
def _non_kolmogorov_psd(self, f: np.ndarray, r0: float) -> np.ndarray:
    # Non-Kolmogorov spectrum with custom spectral index
    beta = self.spectral_index  # Spectral power-law index
    
    # Generalized PSD formula
    psd = 0.023 * r0**(-5/3) * f**(-beta)
    
    return psd
```

## 3. Multi-Layer Atmospheric Modeling

Real atmospheric turbulence varies with altitude. The multi-layer model divides the propagation path into multiple layers, each with its own turbulence strength profile based on the Hufnagel-Valley model:

```python
def _generate_multi_layer_phase_screen(self, r0: float, total_distance: float) -> np.ndarray:
    # Define layer distances
    layer_distances = np.linspace(0, total_distance, self.turbulence_layers + 1)[1:]
    
    # Define Cn2 profile based on altitude (Hufnagel-Valley model)
    cn2_profile = self._calculate_cn2_profile(layer_distances)
    
    # Generate and combine phase screens from each layer
    # ...
    
    return combined_phase_screen
```

## 4. Enhanced Aperture Averaging

Aperture averaging reduces scintillation effects for finite receiver apertures. The enhanced implementation includes:

1. Mode-dependent aperture averaging (higher OAM modes are affected differently)
2. Support for both weak and strong turbulence regimes
3. Inner/outer scale corrections to the aperture averaging factor

```python
def _calculate_enhanced_aperture_averaging_factor(self) -> float:
    # Get basic parameters
    D = self.receiver_aperture_diameter  # Aperture diameter
    r0 = self.r0_current  # Fried parameter
    
    # Calculate normalized aperture diameter
    aperture_ratio = D / r0
    
    # Apply Andrews & Phillips model for weak/strong turbulence
    # ...
    
    # Apply inner/outer scale corrections
    # ...
    
    return factor
```

## 5. Inner/Outer Scale Turbulence Effects

The von Karman spectrum extends the Kolmogorov spectrum by including inner and outer scale effects:

- Inner scale (l₀): Smallest scale of turbulent eddies (typically millimeters)
- Outer scale (L₀): Largest scale of turbulent eddies (typically meters to kilometers)

```python
def _apply_scale_limits(self, f: np.ndarray, psd: np.ndarray) -> np.ndarray:
    if self.inner_scale > 0:
        # von Karman model for inner scale
        fm = 5.92 / (2 * np.pi * self.inner_scale)
        inner_scale_factor = np.exp(-(f/fm)**2)
        psd = psd * inner_scale_factor
        
    if self.outer_scale < float('inf'):
        # von Karman model for outer scale
        f0 = 1.0 / (2 * np.pi * self.outer_scale)
        outer_scale_factor = (1 + (f/f0)**2)**(-self.spectral_index/2)
        psd = psd * outer_scale_factor
        
    return psd
```

## Configuration

The advanced physics features can be configured in the `advanced_physics` section of the configuration file:

```yaml
advanced_physics:
  use_fft_phase_screen: true
  phase_screen_resolution: 256
  phase_screen_size: 2.0
  kolmogorov_spectrum: true
  spectral_index: 3.6667  # 11/3 for Kolmogorov
  inner_scale: 0.001
  outer_scale: 100.0
  turbulence_layers: 3
```

## References

1. Lane, R. G., Glindemann, A., & Dainty, J. C. (1992). Simulation of a Kolmogorov phase screen. Waves in Random Media, 2(3), 209-224.
2. Andrews, L. C., & Phillips, R. L. (2005). Laser beam propagation through random media. SPIE Press.
3. Schmidt, J. D. (2010). Numerical simulation of optical wave propagation with examples in MATLAB. SPIE Press.
4. Fried, D. L. (1965). Statistics of a geometric representation of wavefront distortion. JOSA, 55(11), 1427-1435.
5. Hufnagel, R. E., & Stanley, N. R. (1964). Modulation transfer function associated with image transmission through turbulent media. JOSA, 54(1), 52-61. 