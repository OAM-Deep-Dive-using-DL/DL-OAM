import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import scipy.special as sp
from scipy.constants import c as speed_of_light


class ChannelSimulator:
    """
    High-fidelity physics simulator for OAM wireless channels.
    
    Simulates various physical impairments that affect OAM mode transmission:
    - Path loss
    - Atmospheric turbulence
    - Crosstalk between OAM modes
    - Rician fading
    - Pointing errors
    - Atmospheric attenuation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the channel simulator with configuration parameters.
        
        Args:
            config: Dictionary containing simulation parameters
        """
        # Default parameters
        self.frequency = 28.0e9  # 28 GHz (mmWave)
        self.wavelength = speed_of_light / self.frequency
        self.tx_power_dBm = 30.0  # dBm
        self.noise_figure_dB = 8.0  # 8 dB (updated for more realistic mmWave)
        self.noise_temp = 290.0  # K
        self.bandwidth = 400e6  # 400 MHz
        
        # OAM parameters
        self.min_mode = 1
        self.max_mode = 8
        self.mode_spacing = 1
        self.beam_width = 0.03  # 30 mrad (more realistic)
        
        # Environment parameters  
        self.turbulence_strength = 2.0e-14  # More realistic moderate turbulence Cn^2
        self.pointing_error_std = 0.005  # 5 mrad (more realistic)
        self.rician_k_factor = 8.0  # 8 dB (updated)
        
        # Additional parameters for enhanced modeling
        self.antenna_efficiency = 0.75  # 75% efficiency (realistic for mmWave)
        self.implementation_loss_dB = 3.0  # 3 dB losses (more realistic)
        self.atmospheric_noise_temp = 8.0  # 8 K atmospheric noise
        self.quantum_noise_temp = 0.05  # Reduced quantum noise
        self.receiver_aperture_diameter = 0.3  # 30 cm aperture (more realistic)
        
        # Advanced physics parameters (Phase 6)
        self.use_fft_phase_screen = True  # Use FFT-based phase screen generation
        self.phase_screen_resolution = 256  # Resolution of phase screens
        self.phase_screen_size = 2.0  # Size of phase screen in meters
        self.inner_scale = 0.002  # 2 mm inner scale (more realistic)
        self.outer_scale = 50.0  # 50 m outer scale (more realistic)
        self.turbulence_layers = 3  # 3-layer atmospheric model
        self.kolmogorov_spectrum = True  # Use Kolmogorov spectrum
        self.spectral_index = 11/3  # 11/3 for Kolmogorov
        
        # Update with provided config
        if config:
            self._update_config(config)
        
        # Validate parameters
        self._validate_parameters()
        
        # Derived parameters
        self.tx_power_W = 10 ** (self.tx_power_dBm / 10) / 1000  # Convert dBm to W
        self.k = 2 * np.pi / self.wavelength  # Wave number
        self.num_modes = self.max_mode - self.min_mode + 1
        
        # Initialize channel matrix
        self.H = np.eye(self.num_modes, dtype=complex)
        
        # Initialize phase screen cache for reuse
        self.phase_screen_cache = {}
        
    def clear_phase_screen_cache(self):
        """Clear the phase screen cache to force regeneration."""
        self.phase_screen_cache = {}
        print("Phase screen cache cleared")
        
    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Update simulator parameters from configuration with proper type conversion.
        
        Args:
            config: Dictionary containing simulation parameters
        """
        # Update system parameters
        if 'system' in config:
            system_config = config['system']
            if 'frequency' in system_config:
                self.frequency = float(system_config['frequency'])
                self.wavelength = speed_of_light / self.frequency
            if 'bandwidth' in system_config:
                self.bandwidth = float(system_config['bandwidth'])
            if 'tx_power_dBm' in system_config:
                self.tx_power_dBm = float(system_config['tx_power_dBm'])
            if 'noise_figure_dB' in system_config:
                self.noise_figure_dB = float(system_config['noise_figure_dB'])
            if 'noise_temp' in system_config:
                self.noise_temp = float(system_config['noise_temp'])
        
        # Update OAM parameters
        if 'oam' in config:
            oam_config = config['oam']
            if 'min_mode' in oam_config:
                self.min_mode = int(oam_config['min_mode'])
            if 'max_mode' in oam_config:
                self.max_mode = int(oam_config['max_mode'])
            if 'beam_width' in oam_config:
                self.beam_width = float(oam_config['beam_width'])
        
        # Update environment parameters
        if 'environment' in config:
            env_config = config['environment']
            if 'turbulence_strength' in env_config:
                self.turbulence_strength = float(env_config['turbulence_strength'])
            if 'pointing_error_std' in env_config:
                self.pointing_error_std = float(env_config['pointing_error_std'])
            if 'rician_k_factor' in env_config:
                self.rician_k_factor = float(env_config['rician_k_factor'])
        
        # Update enhanced parameters
        if 'enhanced_params' in config:
            enhanced_config = config['enhanced_params']
            if 'antenna_efficiency' in enhanced_config:
                self.antenna_efficiency = float(enhanced_config['antenna_efficiency'])
            if 'implementation_loss_dB' in enhanced_config:
                self.implementation_loss_dB = float(enhanced_config['implementation_loss_dB'])
            if 'atmospheric_noise_temp' in enhanced_config:
                self.atmospheric_noise_temp = float(enhanced_config['atmospheric_noise_temp'])
            if 'quantum_noise_temp' in enhanced_config:
                self.quantum_noise_temp = float(enhanced_config['quantum_noise_temp'])
            if 'receiver_aperture_diameter' in enhanced_config:
                self.receiver_aperture_diameter = float(enhanced_config['receiver_aperture_diameter'])
        
        # Update advanced physics parameters
        if 'advanced_physics' in config:
            advanced_config = config['advanced_physics']
            if 'use_fft_phase_screen' in advanced_config:
                self.use_fft_phase_screen = bool(advanced_config['use_fft_phase_screen'])
            if 'phase_screen_resolution' in advanced_config:
                self.phase_screen_resolution = int(advanced_config['phase_screen_resolution'])
            if 'phase_screen_size' in advanced_config:
                self.phase_screen_size = float(advanced_config['phase_screen_size'])
            if 'inner_scale' in advanced_config:
                self.inner_scale = float(advanced_config['inner_scale'])
            if 'outer_scale' in advanced_config:
                self.outer_scale = float(advanced_config['outer_scale'])
            if 'turbulence_layers' in advanced_config:
                self.turbulence_layers = int(advanced_config['turbulence_layers'])
            if 'kolmogorov_spectrum' in advanced_config:
                self.kolmogorov_spectrum = bool(advanced_config['kolmogorov_spectrum'])
            if 'spectral_index' in advanced_config:
                self.spectral_index = float(advanced_config['spectral_index'])
        
        # Clear cache when configuration changes
        self.clear_phase_screen_cache()
    
    def _validate_parameters(self):
        """Validate all simulation parameters are within reasonable ranges."""
        
        # Frequency validation
        if not (1e9 <= self.frequency <= 1000e9):  # 1 GHz to 1 THz
            raise ValueError(f"Frequency {self.frequency/1e9:.1f} GHz is outside reasonable range (1-1000 GHz)")
        
        # Power validation
        if not (-20 <= self.tx_power_dBm <= 50):
            raise ValueError(f"TX power {self.tx_power_dBm} dBm is outside reasonable range (-20 to 50 dBm)")
        
        # Noise figure validation
        if not (0 <= self.noise_figure_dB <= 20):
            raise ValueError(f"Noise figure {self.noise_figure_dB} dB is outside reasonable range (0-20 dB)")
        
        # Temperature validation
        if not (50 <= self.noise_temp <= 500):
            raise ValueError(f"Noise temperature {self.noise_temp} K is outside reasonable range (50-500 K)")
        
        # Bandwidth validation
        if not (1e6 <= self.bandwidth <= 10e9):  # 1 MHz to 10 GHz
            raise ValueError(f"Bandwidth {self.bandwidth/1e6:.1f} MHz is outside reasonable range (1-10000 MHz)")
        
        # OAM mode validation
        if not (1 <= self.min_mode < self.max_mode <= 20):
            raise ValueError(f"OAM modes [{self.min_mode}, {self.max_mode}] are outside reasonable range")
        
        # Beam width validation
        if not (0.001 <= self.beam_width <= 1.0):  # 1 mrad to 1 rad
            raise ValueError(f"Beam width {self.beam_width} rad is outside reasonable range (0.001-1.0 rad)")
        
        # Turbulence strength validation
        if not (1e-18 <= self.turbulence_strength <= 1e-10):
            raise ValueError(f"Turbulence strength {self.turbulence_strength} is outside reasonable range")
        
        # Pointing error validation
        if not (0.0001 <= self.pointing_error_std <= 0.1):  # 0.1 mrad to 100 mrad
            raise ValueError(f"Pointing error {self.pointing_error_std} rad is outside reasonable range")
        
        # Efficiency validation
        if not (0.1 <= self.antenna_efficiency <= 1.0):
            raise ValueError(f"Antenna efficiency {self.antenna_efficiency} is outside reasonable range (0.1-1.0)")
        
        # Advanced physics parameters validation
        if not (32 <= self.phase_screen_resolution <= 2048):
            raise ValueError(f"Phase screen resolution {self.phase_screen_resolution} is outside reasonable range (32-2048)")
        
        if not (0.0001 <= self.inner_scale <= 0.1):
            raise ValueError(f"Inner scale {self.inner_scale} m is outside reasonable range (0.0001-0.1 m)")
            
        if not (10.0 <= self.outer_scale <= 1000.0):
            raise ValueError(f"Outer scale {self.outer_scale} m is outside reasonable range (10-1000 m)")
            
        if not (1 <= self.turbulence_layers <= 10):
            raise ValueError(f"Turbulence layers {self.turbulence_layers} is outside reasonable range (1-10)")
            
        if not (2.0 <= self.spectral_index <= 4.0):
            raise ValueError(f"Spectral index {self.spectral_index} is outside reasonable range (2.0-4.0)")
        
        print("✅ All parameters validated successfully")

    def _calculate_path_loss(self, distance: float) -> float:
        """
        Calculate free space path loss.
        
        Args:
            distance: Distance between transmitter and receiver in meters
            
        Returns:
            Path loss in linear scale
        """
        # Free space path loss formula: (4πd/λ)^2
        path_loss_linear = (4 * np.pi * distance / self.wavelength) ** 2
        return path_loss_linear
    
    def _generate_turbulence_screen(self, distance: float) -> np.ndarray:
        """
        Generate atmospheric turbulence phase screen.
        
        Args:
            distance: Propagation distance in meters
            
        Returns:
            Phase perturbation matrix for each OAM mode
        """
        # Calculate Fried parameter (r0) - CORRECT FORMULA FROM LITERATURE
        # Wikipedia reference: r0 = [0.423 * k^2 * ∫ Cn²(z') dz']^(-3/5)
        # For uniform turbulence: ∫ Cn²(z') dz' = Cn² * distance
        cn2_integral = self.turbulence_strength * distance
        r0 = (0.423 * (self.k**2) * cn2_integral) ** (-3/5)
        
        self.r0_current = r0  # Store for other methods
        
        # Calculate beam wandering variance
        beam_wander_variance = self._calculate_beam_wander(distance, self.turbulence_strength)
        
        # Calculate scintillation index
        scintillation_index = self._calculate_scintillation_index(distance, self.turbulence_strength)
        
        # Generate phase screen using either FFT-based or direct method
        if self.use_fft_phase_screen:
            # Use advanced FFT-based phase screen generation
            if self.turbulence_layers > 1:
                # Multi-layer atmospheric modeling
                phase_screen = self._generate_multi_layer_phase_screen(r0, distance)
            else:
                # Single layer modeling
                phase_screen = self._generate_fft_phase_screen(r0, distance)
            
            # Apply phase screen to OAM modes
            phase_perturbations = self._apply_phase_screen_to_modes(phase_screen, r0, scintillation_index, beam_wander_variance)
        else:
            # Use original direct method for backward compatibility
            phase_perturbations = self._generate_direct_phase_screen(r0, scintillation_index, beam_wander_variance)
        
        return phase_perturbations
        
    def _generate_multi_layer_phase_screen(self, r0: float, total_distance: float) -> np.ndarray:
        """
        Generate multi-layer atmospheric phase screen.
        
        Args:
            r0: Fried parameter for the total path
            total_distance: Total propagation distance
            
        Returns:
            Combined phase screen from multiple layers
        """
        # Check if we have a cached multi-layer phase screen for this r0 and distance
        cache_key = f"multi_{r0:.6f}_{total_distance:.1f}_{self.turbulence_layers}"
        if cache_key in self.phase_screen_cache:
            return self.phase_screen_cache[cache_key]
            
        # Define layer distances and Cn2 profile
        layer_distances = np.linspace(0, total_distance, self.turbulence_layers + 1)[1:]
        
        # Define Cn2 profile based on altitude
        # Using Hufnagel-Valley model for Cn2 distribution with altitude
        cn2_profile = self._calculate_cn2_profile(layer_distances)
        
        # Calculate layer weights based on Cn2 profile
        total_cn2 = np.sum(cn2_profile)
        layer_weights = cn2_profile / total_cn2 if total_cn2 > 0 else np.ones_like(cn2_profile) / len(cn2_profile)
        
        # Generate phase screen for each layer and combine
        combined_phase_screen = np.zeros((self.phase_screen_resolution, self.phase_screen_resolution))
        
        for i, (distance, weight) in enumerate(zip(layer_distances, layer_weights)):
            # Calculate effective r0 for this layer
            layer_r0 = r0 * (weight ** (-3/5))
            
            # Generate phase screen for this layer
            layer_screen = self._generate_fft_phase_screen(layer_r0, distance)
            
            # Add to combined screen with appropriate weight
            combined_phase_screen += layer_screen * np.sqrt(weight)
        
        # Cache the result
        self.phase_screen_cache[cache_key] = combined_phase_screen
        
        return combined_phase_screen
    
    def _calculate_cn2_profile(self, distances: np.ndarray) -> np.ndarray:
        """
        Calculate Cn2 profile based on Hufnagel-Valley model.
        
        Args:
            distances: Array of distances from transmitter
            
        Returns:
            Array of Cn2 values for each distance
        """
        # Convert distances to altitudes (assuming ground-to-air or air-to-ground link)
        # This is a simplified model - in reality, the path geometry would be more complex
        max_altitude = 20000  # meters
        altitudes = distances * max_altitude / np.max(distances) if np.max(distances) > 0 else np.zeros_like(distances)
        
        # Hufnagel-Valley Cn2 profile
        cn2_profile = np.zeros_like(altitudes)
        
        for i, h in enumerate(altitudes):
            if h < 1000:  # Below 1 km
                # Strong turbulence near ground
                cn2_profile[i] = self.turbulence_strength * np.exp(-h/100)
            elif h < 6000:  # 1-6 km (troposphere)
                # Moderate turbulence in troposphere
                cn2_profile[i] = self.turbulence_strength * 0.1 * np.exp(-h/1000)
            else:  # Above 6 km (stratosphere)
                # Weak turbulence in stratosphere
                cn2_profile[i] = self.turbulence_strength * 0.01 * np.exp(-h/5000)
        
        return cn2_profile
    
    def _generate_direct_phase_screen(self, r0: float, scintillation_index: float, beam_wander_variance: float) -> np.ndarray:
        """
        Generate phase screen using direct method (original implementation).
        
        Args:
            r0: Fried parameter
            scintillation_index: Scintillation index
            beam_wander_variance: Beam wandering variance
            
        Returns:
            Phase perturbation matrix for each OAM mode
        """
        # Generate random phase perturbations for each mode
        phase_perturbations = np.zeros((self.num_modes, self.num_modes), dtype=complex)
        
        for i in range(self.num_modes):
            mode_i = i + self.min_mode
            for j in range(self.num_modes):
                mode_j = j + self.min_mode
                
                # Phase variance increases with mode number difference
                if i == j:
                    # Diagonal elements (same mode)
                    # Include scintillation effects
                    variance = (mode_i / max(r0, 1e-10)) ** (5/3)
                    variance *= (1 + scintillation_index)  # Add scintillation
                else:
                    # Off-diagonal elements (crosstalk)
                    # Enhanced coupling model based on mode overlap
                    r0_safe = max(r0, 1e-10)
                    mode_diff = abs(mode_i - mode_j)
                    
                    # Improved coupling based on Laguerre-Gaussian mode overlap
                    coupling_strength = self._calculate_mode_coupling(mode_i, mode_j, r0_safe, 0)
                    variance = coupling_strength * (1.0) ** (5/6)  # Base dependency
                
                # Add beam wandering effect to phase variance
                if i == j:
                    variance += beam_wander_variance
                
                # Generate random phase with improved distribution
                phase = np.random.normal(0, np.sqrt(max(variance, 1e-15)))
                amplitude = 1.0
                
                # Add amplitude fluctuations due to scintillation
                if i == j:
                    amplitude_var = scintillation_index * 0.5
                    amplitude *= np.exp(np.random.normal(0, np.sqrt(amplitude_var)))
                
                phase_perturbations[i, j] = amplitude * np.exp(1j * phase)
        
        return phase_perturbations
    
    def _generate_fft_phase_screen(self, r0: float, distance: float) -> np.ndarray:
        """
        Generate atmospheric turbulence phase screen using FFT-based method (McGlamery).
        
        Args:
            r0: Fried parameter (coherence length)
            distance: Propagation distance
            
        Returns:
            2D phase screen array
        """
        # Check if we have a cached phase screen for this r0 and distance
        cache_key = f"{r0:.6f}_{distance:.1f}_{self.turbulence_strength:.2e}"
        if cache_key in self.phase_screen_cache:
            return self.phase_screen_cache[cache_key]
            
        # Set up grid parameters
        N = self.phase_screen_resolution  # Grid size
        L = self.phase_screen_size  # Physical size (meters)
        
        # Compute spatial frequencies
        df = 1.0 / L  # Frequency grid spacing
        fx = np.arange(-N/2, N/2) * df
        fy = fx.copy()
        fx, fy = np.meshgrid(fx, fy)
        
        # Compute radial frequency
        f = np.sqrt(fx**2 + fy**2)
        
        # Avoid division by zero at origin
        f[N//2, N//2] = df  # Set to minimum non-zero frequency
        
        # Create power spectrum based on selected model
        if self.kolmogorov_spectrum:
            # Standard Kolmogorov spectrum
            psd = self._kolmogorov_psd(f, r0)
        else:
            # Non-Kolmogorov spectrum with custom spectral index
            psd = self._non_kolmogorov_psd(f, r0)
        
        # Apply inner and outer scale modifications if specified
        psd = self._apply_scale_limits(f, psd)
        
        # Generate complex Gaussian random field
        real_part = np.random.normal(0, 1, (N, N))
        imag_part = np.random.normal(0, 1, (N, N))
        
        # Create complex amplitude field
        # For FFT method: amplitude = sqrt(PSD * df^2)
        amplitude = np.sqrt(np.maximum(psd, 0) * df**2)
        
        # Create complex field
        complex_field = (real_part + 1j * imag_part) * amplitude
        
        # Ensure DC component is zero for zero-mean phase screen
        complex_field[N//2, N//2] = 0.0
        
        # Perform inverse FFT to get phase screen
        phase_screen = np.real(np.fft.ifft2(np.fft.ifftshift(complex_field)))
        
        # Remove any DC offset to ensure zero mean
        phase_screen = phase_screen - np.mean(phase_screen)
        
        # Scale to achieve realistic phase variance based on r0 and turbulence strength
        current_var = np.var(phase_screen)
        
        if current_var > 1e-15:  # If we have some variance from FFT
            # For mmWave OAM systems, use a different scaling approach
            # since r0 values are very large compared to beam sizes
            
            # Direct scaling based on Cn2 and distance for mmWave systems
            # This accounts for the fact that mmWave beams are less affected by 
            # small-scale turbulence but more by large-scale refractive effects
            
            # Base variance scales with Cn2 and distance
            cn2_normalized = self.turbulence_strength / 1e-15  # Normalize to typical clear air
            distance_km = distance / 1000.0
            
            # Empirical model for mmWave OAM phase variance
            # Accounts for the different sensitivity compared to optical wavelengths
            base_variance = 0.01 * cn2_normalized ** 0.7 * distance_km ** 0.5
            
            # Scale by frequency (higher frequencies more sensitive to large-scale effects)
            freq_factor = (self.frequency / 28e9) ** 0.3
            
            # Scale by OAM mode order (higher modes more sensitive)
            # Estimate average mode from the simulation setup
            avg_mode = (self.min_mode + self.max_mode) / 2.0
            mode_factor = 1.0 + 0.1 * (avg_mode - 1)
            
            # Combined target variance
            target_var = base_variance * freq_factor * mode_factor
            
            # Ensure reasonable bounds for mmWave systems
            target_var = np.clip(target_var, 0.001, 3.0)
            
            # Scale to target variance
            scale_factor = np.sqrt(target_var / current_var)
            phase_screen = phase_screen * scale_factor
            
        else:
            # If no variance from FFT, create turbulence based on Cn2 directly
            cn2_normalized = self.turbulence_strength / 1e-15
            target_std = 0.05 * (cn2_normalized ** 0.35)  # Square root of variance scaling
            target_std = np.clip(target_std, 0.01, 1.5)  # Reasonable bounds
            phase_screen = np.random.normal(0, target_std, (N, N))
        
        # Final check to ensure zero mean
        phase_screen = phase_screen - np.mean(phase_screen)
        
        # Cache the result for reuse
        self.phase_screen_cache[cache_key] = phase_screen
        
        return phase_screen
    
    def _kolmogorov_psd(self, f: np.ndarray, r0: float) -> np.ndarray:
        """
        Compute Kolmogorov power spectral density.
        
        Args:
            f: Spatial frequency grid
            r0: Fried parameter
            
        Returns:
            Power spectral density array
        """
        # Standard Kolmogorov spectrum formula for atmospheric phase
        # PSD(f) = 0.023 * r0^(-5/3) * f^(-11/3) 
        # This is the correct form in the inertial range
        
        # Avoid division by zero - set minimum frequency
        f_safe = np.maximum(f, 1e-10)
        
        # Standard Kolmogorov spectrum
        psd = 0.023 * (r0 ** (-5/3)) * (f_safe ** (-11/3))
        
        # Handle zero frequency specially - use nearby value
        center = self.phase_screen_resolution // 2
        if f[center, center] == 0 or f[center, center] < 1e-10:
            # Use value from neighboring point for DC component
            if center + 1 < self.phase_screen_resolution:
                psd[center, center] = psd[center, center + 1]
            else:
                psd[center, center] = np.mean(psd[center-1:center+2, center-1:center+2])
        
        # Ensure no infinite or NaN values
        psd = np.where(np.isfinite(psd), psd, 0)
        
        return psd
    
    def _non_kolmogorov_psd(self, f: np.ndarray, r0: float) -> np.ndarray:
        """
        Compute non-Kolmogorov power spectral density with custom spectral index.
        
        Args:
            f: Spatial frequency grid
            r0: Fried parameter
            
        Returns:
            Power spectral density array
        """
        # Non-Kolmogorov spectrum with custom spectral index
        beta = self.spectral_index  # Spectral power-law index
        
        # Generalized PSD formula with proper scaling
        # For non-Kolmogorov, we maintain the same r0 dependency but change the spectral slope
        psd = 0.023 * (r0 ** (-5/3)) * (f ** (-beta))
        
        # Handle origin (zero frequency) properly
        center = self.phase_screen_resolution // 2
        psd[center, center] = psd[center, center+1] if center+1 < self.phase_screen_resolution else 0
        
        # Ensure no infinite or NaN values
        psd = np.where(np.isfinite(psd), psd, 0)
        
        return psd
    
    def _apply_scale_limits(self, f: np.ndarray, psd: np.ndarray) -> np.ndarray:
        """
        Apply inner and outer scale modifications to the power spectral density.
        
        Args:
            f: Spatial frequency grid
            psd: Power spectral density
            
        Returns:
            Modified power spectral density
        """
        modified_psd = psd.copy()
        
        if self.inner_scale > 0:
            # von Karman model for inner scale
            # High-frequency rolloff due to viscous dissipation
            fm = 5.92 / (2 * np.pi * self.inner_scale)  # Inner scale frequency
            inner_scale_factor = np.exp(-(f/fm)**2)
            modified_psd = modified_psd * inner_scale_factor
            
        if self.outer_scale < float('inf') and self.outer_scale > 0:
            # von Karman model for outer scale  
            # Low-frequency rolloff due to finite energy injection scale
            f0 = 1.0 / (2 * np.pi * self.outer_scale)  # Outer scale frequency
            
            # More gradual outer scale rolloff to prevent severe PSD reduction
            # Use a modified von Karman form that doesn't go to zero
            outer_scale_factor = (1 + (f/f0)**2)**(-11/12)
            
            # Prevent factor from going below a minimum threshold
            min_factor = 1e-6  # Minimum allowable factor
            outer_scale_factor = np.maximum(outer_scale_factor, min_factor)
            
            # Apply only to low frequencies (below injection scale)
            # For frequencies much higher than f0, keep original PSD
            frequency_mask = f < 10 * f0  # Apply outer scale only up to 10x the outer scale frequency
            outer_scale_factor = np.where(frequency_mask, outer_scale_factor, 1.0)
            
            modified_psd = modified_psd * outer_scale_factor
            
        return modified_psd
    
    def _apply_phase_screen_to_modes(self, phase_screen: np.ndarray, r0: float, 
                                    scintillation_index: float, beam_wander_variance: float) -> np.ndarray:
        """
        Apply phase screen to OAM modes.
        
        Args:
            phase_screen: 2D phase screen array
            r0: Fried parameter
            scintillation_index: Scintillation index
            beam_wander_variance: Beam wandering variance
            
        Returns:
            Phase perturbation matrix for each OAM mode
        """
        # Initialize phase perturbation matrix
        phase_perturbations = np.zeros((self.num_modes, self.num_modes), dtype=complex)
        
        # Get screen center and resolution
        N = self.phase_screen_resolution
        center = N // 2
        
        # Calculate grid spacing
        delta = self.phase_screen_size / N
        
        for i in range(self.num_modes):
            mode_i = i + self.min_mode
            
            # Calculate OAM mode radius (larger modes have larger radii)
            mode_radius = int(np.sqrt(mode_i) * N / 8)
            mode_radius = max(5, min(mode_radius, N//4))  # Constrain radius
            
            # Extract phase values along the mode radius
            theta_values = np.linspace(0, 2*np.pi, 36, endpoint=False)
            phase_values = []
            
            for theta in theta_values:
                x = int(center + mode_radius * np.cos(theta))
                y = int(center + mode_radius * np.sin(theta))
                
                # Ensure coordinates are within bounds
                x = max(0, min(x, N-1))
                y = max(0, min(y, N-1))
                
                phase_values.append(phase_screen[y, x])
            
            # Calculate average phase and variance for this mode
            avg_phase = np.mean(phase_values)
            phase_var = np.var(phase_values)
            
            # Apply to diagonal (same mode)
            # Turbulence degrades the channel, so amplitude should decrease with turbulence
            base_amplitude = 1.0
            
            # Phase variance degrades the coherent signal
            if phase_var > 0:
                # Higher phase variance reduces signal coherence
                coherence_loss = np.exp(-phase_var / 2.0)  # Exponential decay with variance
                base_amplitude *= coherence_loss
            
            # Scintillation adds additional amplitude fluctuations
            if scintillation_index > 0:
                # Add both positive and negative fluctuations, but with net degradation
                amplitude_var = scintillation_index * 0.3
                # Use log-normal distribution which can create both fading and enhancement
                # but with proper scaling to ensure net degradation
                scint_factor = np.exp(np.random.normal(-amplitude_var/2, np.sqrt(amplitude_var)))
                base_amplitude *= scint_factor
            
            # Beam wandering effect - reduces effective signal coupling
            if beam_wander_variance > 0:
                # Beam wandering reduces the signal strength
                wander_loss = 1.0 / (1.0 + beam_wander_variance * 1000)  # Scaling factor
                base_amplitude *= wander_loss
            
            phase_perturbations[i, i] = base_amplitude * np.exp(1j * avg_phase)
            
            # Calculate cross-coupling to other modes
            for j in range(self.num_modes):
                if i == j:
                    continue
                    
                mode_j = j + self.min_mode
                mode_diff = abs(mode_i - mode_j)
                
                # Calculate coupling based on phase screen statistics and mode difference
                coupling_strength = self._calculate_mode_coupling(mode_i, mode_j, r0, 0)
                
                # Generate correlated phase
                correlated_phase = avg_phase * (1.0 - 0.1 * mode_diff) + np.random.normal(0, phase_var * 0.5)
                
                # Apply coupling
                phase_perturbations[i, j] = coupling_strength * np.exp(1j * correlated_phase)
        
        return phase_perturbations
    
    def _calculate_beam_wander(self, distance: float, cn2: float) -> float:
        """
        Calculate beam wandering variance.
        
        Args:
            distance: Propagation distance in meters
            cn2: Turbulence strength parameter
            
        Returns:
            Beam wandering variance in radians^2
        """
        # CORRECTED: Beam wandering formula from Andrews & Phillips
        # Formula: σ_w^2 = 2.42 * Cn^2 * k^2 * L^3 (for plane wave, point receiver)
        # Previous formula had wrong exponents
        sigma_r2 = 2.42 * cn2 * (self.k ** 2) * (distance ** 3)
        return sigma_r2
    
    def _calculate_scintillation_index(self, distance: float, cn2: float) -> float:
        """
        Calculate scintillation index for weak turbulence.
        
        Args:
            distance: Propagation distance in meters
            cn2: Turbulence strength parameter
            
        Returns:
            Scintillation index (dimensionless)
        """
        # Rytov variance for plane wave: sigma_I^2 = 1.23 * Cn^2 * k^(7/6) * L^(11/6)
        rytov_variance = 1.23 * cn2 * (self.k ** (7/6)) * (distance ** (11/6))
        
        # For weak turbulence, scintillation index ≈ Rytov variance
        # For moderate turbulence, use modified formula
        if rytov_variance < 1.0:
            scintillation_index = rytov_variance
        else:
            # Moderate to strong turbulence regime
            scintillation_index = rytov_variance / (1 + rytov_variance)
            
        return min(scintillation_index, 2.0)  # Cap at reasonable value
    
    def _calculate_mode_coupling(self, mode_l: int, mode_m: int, r0: float, distance: float) -> float:
        """
        Calculate OAM mode coupling coefficient based on atmospheric distortion.
        
        Args:
            mode_l: First OAM mode number
            mode_m: Second OAM mode number  
            r0: Fried parameter
            distance: Propagation distance
            
        Returns:
            Coupling coefficient
        """
        if mode_l == mode_m:
            return 1.0
            
        mode_diff = abs(mode_l - mode_m)
        
        # Physics-based coupling model for OAM modes
        # Based on Laguerre-Gaussian beam overlap integrals under atmospheric turbulence
        
        # Base coupling strengths based on modal selection rules
        # Strongest coupling occurs for Δl = ±1, ±2 due to turbulence-induced mode mixing
        if mode_diff == 1:
            base_coupling = 0.15  # Adjacent mode coupling
        elif mode_diff == 2:
            base_coupling = 0.08  # Second-order coupling
        elif mode_diff == 3:
            base_coupling = 0.04  # Third-order coupling
        else:
            base_coupling = 0.01 * np.exp(-mode_diff / 4.0)  # Exponential decay for higher orders
        
        # CRITICAL FIX: Turbulence-induced coupling enhancement
        # Coupling should increase with stronger turbulence (smaller r0)
        # Use D/r0 ratio where D is characteristic beam size
        beam_diameter = 0.5  # Characteristic beam size (meters)
        turbulence_ratio = beam_diameter / max(r0, 0.001)  # Prevent division by zero
        
        # Coupling enhancement follows theoretical scaling
        if turbulence_ratio < 0.1:  # Very weak turbulence
            turbulence_factor = 1.0 + 0.1 * turbulence_ratio
        elif turbulence_ratio < 1.0:  # Weak to moderate turbulence
            turbulence_factor = 1.0 + 0.2 * turbulence_ratio**0.5
        elif turbulence_ratio < 10.0:  # Moderate to strong turbulence  
            turbulence_factor = 1.0 + 0.5 * np.log10(turbulence_ratio)
        else:  # Very strong turbulence
            turbulence_factor = 1.0 + 1.0 + 0.2 * np.log10(turbulence_ratio / 10)
        
        # Distance factor (weaker effect)
        distance_factor = 1.0 + 0.05 * np.log10(max(distance / 1000.0, 1.0))
        
        # Calculate final coupling
        coupling = base_coupling * turbulence_factor * distance_factor
        
        # Apply physical limits
        max_coupling = 0.4 if mode_diff <= 2 else 0.2
        return min(coupling, max_coupling)
    
    def _calculate_crosstalk(self, distorted_field: np.ndarray) -> np.ndarray:
        """
        Calculate crosstalk between OAM modes due to distortions.
        
        Args:
            distorted_field: Distorted field matrix from turbulence
            
        Returns:
            Enhanced crosstalk matrix with proper mode coupling
        """
        # Initialize enhanced crosstalk matrix
        crosstalk_matrix = np.zeros((self.num_modes, self.num_modes), dtype=complex)
        
        # Apply distorted field first
        base_matrix = distorted_field.copy()
        
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                if i == j:
                    # Same mode (diagonal elements) - include field distortion
                    crosstalk_matrix[i, j] = base_matrix[i, j]
                else:
                    # Different modes (off-diagonal elements)
                    # Use the enhanced coupling from distorted field
                    crosstalk_matrix[i, j] = base_matrix[i, j]
        
        # Apply aperture averaging if receiver aperture is defined
        if hasattr(self, 'receiver_aperture_diameter'):
            crosstalk_matrix = self._apply_aperture_averaging(crosstalk_matrix)
        
        # Normalize to ensure energy conservation with improved method
        for i in range(self.num_modes):
            # Calculate total power in row
            total_power = np.sum(np.abs(crosstalk_matrix[i, :])**2)
            if total_power > 1e-15:
                # Normalize but preserve relative coupling strengths
                crosstalk_matrix[i, :] = crosstalk_matrix[i, :] / np.sqrt(total_power)
        
        return crosstalk_matrix
    
    def _apply_aperture_averaging(self, channel_matrix: np.ndarray) -> np.ndarray:
        """
        Apply enhanced aperture averaging effects to reduce scintillation.
        
        Args:
            channel_matrix: Input channel matrix
            
        Returns:
            Channel matrix with aperture averaging applied
        """
        if not hasattr(self, 'receiver_aperture_diameter'):
            return channel_matrix
            
        # Calculate enhanced aperture averaging factor
        aperture_factor = self._calculate_enhanced_aperture_averaging_factor()
        
        # Apply averaging to diagonal elements (reduces scintillation)
        averaged_matrix = channel_matrix.copy()
        
        # Enhanced aperture averaging with mode-dependent effects
        for i in range(self.num_modes):
            mode_i = i + self.min_mode
            
            # Higher OAM modes are affected differently by aperture averaging
            # due to their larger spatial extent
            mode_factor = 1.0 + 0.05 * (mode_i - self.min_mode)  # Higher modes get more averaging
            
            # Reduce amplitude fluctuations (scintillation)
            amplitude = np.abs(averaged_matrix[i, i])
            phase = np.angle(averaged_matrix[i, i])
            
            # Apply enhanced aperture averaging to amplitude
            # The effect is stronger for higher turbulence and larger apertures
            averaged_amplitude = amplitude * (aperture_factor ** (1.0 / mode_factor))
            averaged_matrix[i, i] = averaged_amplitude * np.exp(1j * phase)
            
            # Also apply to off-diagonal elements (crosstalk reduction)
            for j in range(self.num_modes):
                if i == j:
                    continue
                
                # Crosstalk reduction due to aperture averaging
                # Larger apertures reduce crosstalk between modes
                crosstalk_reduction = 1.0 - 0.2 * (1.0 - aperture_factor)
                averaged_matrix[i, j] *= crosstalk_reduction
            
        return averaged_matrix
    
    def _calculate_enhanced_aperture_averaging_factor(self) -> float:
        """
        Calculate enhanced aperture averaging factor with inner/outer scale effects.
        
        Returns:
            Enhanced aperture averaging factor (between 0 and 1)
        """
        if not hasattr(self, 'receiver_aperture_diameter') or not hasattr(self, 'r0_current'):
            return 1.0
            
        # Get basic parameters
        D = self.receiver_aperture_diameter  # Aperture diameter
        r0 = max(getattr(self, 'r0_current', 0.1), 1e-10)  # Fried parameter
        k = self.k  # Wave number
        
        # Calculate normalized aperture diameter
        aperture_ratio = D / r0
        
        # Calculate Rytov variance (measure of turbulence strength)
        # This is a simplified version - in a real system, this would be measured
        rytov_variance = getattr(self, 'last_scintillation_index', 0.5)
        
        # Enhanced aperture averaging formula based on Andrews & Phillips model
        # This accounts for both weak and strong turbulence regimes
        if rytov_variance < 1.0:  # Weak turbulence
            # Andrews & Phillips formula for weak turbulence
            if aperture_ratio < 1.0:
                factor = 1.0 - 0.5 * aperture_ratio**(5/3)
            else:
                factor = 0.5 * aperture_ratio**(-5/3)
        else:  # Strong turbulence
            # Modified formula for strong turbulence
            if aperture_ratio < 1.0:
                factor = 1.0 - 0.4 * aperture_ratio
            else:
                factor = 0.6 + 0.4 / aperture_ratio
        
        # Apply inner/outer scale corrections if enabled
        if hasattr(self, 'inner_scale') and self.inner_scale > 0:
            # Inner scale correction (small-scale effects)
            kl0 = k * self.inner_scale
            inner_correction = 1.0 + 0.4 * np.exp(-(kl0)**2)
            factor *= inner_correction
            
        if hasattr(self, 'outer_scale') and self.outer_scale < float('inf'):
            # Outer scale correction (large-scale effects)
            kL0 = k * self.outer_scale
            if kL0 > 0:
                outer_correction = 1.0 - 0.4 * np.exp(-(D/self.outer_scale)**2)
                factor *= outer_correction
        
        # Ensure factor is within reasonable bounds
        return max(min(factor, 1.0), 0.05)
        
    def _get_rician_fading_gain(self) -> np.ndarray:
        """
        Calculate Rician fading channel gains.
        
        Returns:
            Matrix of Rician fading gains for each mode
        """
        # Convert K-factor from dB to linear
        k_linear = 10 ** (self.rician_k_factor / 10)
        
        # Rician fading parameters
        v = np.sqrt(k_linear / (k_linear + 1))  # LOS component
        sigma = np.sqrt(1 / (2 * (k_linear + 1)))  # Scatter component std
        
        # Generate fading gains for each mode
        fading_matrix = np.zeros((self.num_modes, self.num_modes), dtype=complex)
        
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                if i == j:
                    # Line-of-sight component (diagonal)
                    los = v
                    
                    # Scattered component (complex Gaussian)
                    scatter_real = np.random.normal(0, sigma)
                    scatter_imag = np.random.normal(0, sigma)
                    scatter = scatter_real + 1j * scatter_imag
                    
                    # Combined Rician fading
                    fading_matrix[i, j] = los + scatter
                else:
                    # Only scattered component for off-diagonal (reduced)
                    scatter_real = np.random.normal(0, sigma * 0.1)
                    scatter_imag = np.random.normal(0, sigma * 0.1)
                    fading_matrix[i, j] = scatter_real + 1j * scatter_imag
        
        return fading_matrix
    
    def _get_pointing_error_loss(self, oam_mode: int) -> float:
        """
        Calculate loss due to pointing errors with improved OAM mode sensitivity.
        
        Args:
            oam_mode: Current OAM mode
            
        Returns:
            Pointing error loss in linear scale
        """
        # Generate random pointing error
        pointing_error = np.random.normal(0, self.pointing_error_std)
        
        # Enhanced sensitivity model for OAM modes
        # Higher OAM modes are more sensitive to pointing errors
        # Sensitivity increases approximately linearly with mode number
        base_sensitivity = oam_mode / self.min_mode
        
        # Additional sensitivity factor based on beam characteristics
        # OAM beams have donut-shaped intensity profiles
        beam_sensitivity = 1.0 + 0.1 * oam_mode  # Additional sensitivity
        
        total_sensitivity = base_sensitivity * beam_sensitivity
        
        # Calculate loss using improved Gaussian beam model
        # Include both radial and angular components
        radial_loss = np.exp(-(pointing_error * total_sensitivity)**2 / (2 * self.beam_width**2))
        
        # Angular sensitivity (phase gradient effects)
        angular_sensitivity = 0.1 * oam_mode * abs(pointing_error)
        angular_loss = 1.0 / (1.0 + angular_sensitivity)
        
        # Combined pointing loss
        pointing_loss = radial_loss * angular_loss
        
        return max(pointing_loss, 0.01)  # Minimum 1% transmission
    
    def _get_atmospheric_attenuation(self, distance: float) -> float:
        """
        Calculate atmospheric attenuation.
        
        Args:
            distance: Distance in meters
            
        Returns:
            Atmospheric attenuation in linear scale
        """
        # Simplified model for atmospheric attenuation
        # At mmWave frequencies, attenuation is primarily due to oxygen and water vapor
        
        # Typical attenuation at 28 GHz is around 0.1 dB/km for clear air
        attenuation_dB_per_km = 0.1
        
        # Convert to linear scale
        attenuation_linear = 10 ** (attenuation_dB_per_km * distance / 10000)
        
        return attenuation_linear
    
    def run_step(self, user_position: np.ndarray, current_oam_mode: int) -> Tuple[np.ndarray, float]:
        """
        Run one step of the channel simulation.
        
        Args:
            user_position: 3D position of the user [x, y, z] in meters
            current_oam_mode: Current OAM mode being used
            
        Returns:
            Tuple of (channel matrix H, SINR in dB)
        """
        # Input validation
        if not isinstance(user_position, np.ndarray) or user_position.size != 3:
            raise ValueError("user_position must be a 3D numpy array")
        
        if not (self.min_mode <= current_oam_mode <= self.max_mode):
            raise ValueError(f"current_oam_mode {current_oam_mode} must be between {self.min_mode} and {self.max_mode}")
        
        # Calculate distance from origin (assumed transmitter position)
        distance = np.linalg.norm(user_position)
        
        # Validate distance is reasonable
        if distance < 1.0:
            distance = 1.0  # Minimum distance to avoid singularities
        elif distance > 50000:  # 50 km max
            distance = 50000
        
        # 1. Path loss
        path_loss = self._calculate_path_loss(distance)
        
        # 2. Atmospheric turbulence
        turbulence_screen = self._generate_turbulence_screen(distance)
        
        # Store r0 for aperture averaging calculations - use same formula as turbulence screen
        cn2_integral = self.turbulence_strength * distance
        self.r0_current = (0.423 * (self.k**2) * cn2_integral) ** (-3/5)
        
        # Store scintillation index for aperture averaging
        self.last_scintillation_index = self._calculate_scintillation_index(distance, self.turbulence_strength)
        
        # 3. Crosstalk
        crosstalk_matrix = self._calculate_crosstalk(turbulence_screen)
        
        # 4. Rician fading
        fading_matrix = self._get_rician_fading_gain()
        
        # 5. Pointing error (specific to current mode)
        pointing_loss = self._get_pointing_error_loss(current_oam_mode)
        
        # 6. Atmospheric attenuation
        attenuation = self._get_atmospheric_attenuation(distance)
        
        # Combine all effects to get channel matrix H
        # Add a small epsilon to avoid division by zero
        path_loss = max(path_loss, 1e-10)
        attenuation = max(attenuation, 1e-10)
        
        # Calculate channel gain (inverse of losses)
        # Note: path_loss and attenuation are linear loss factors (>1)
        channel_gain = 1.0 / (path_loss * attenuation)
        
        # Apply antenna efficiency to signal (not to noise floor)
        antenna_efficiency = getattr(self, 'antenna_efficiency', 0.75)
        channel_gain = channel_gain * antenna_efficiency
        
        # Start with crosstalk matrix and apply channel gain
        self.H = crosstalk_matrix * fading_matrix * np.sqrt(channel_gain)
        
        # Apply pointing loss to current mode
        mode_idx = current_oam_mode - self.min_mode
        self.H[mode_idx, :] *= pointing_loss
        self.H[:, mode_idx] *= pointing_loss
        
        # Calculate SINR for current mode
        signal_power = self.tx_power_W * np.abs(self.H[mode_idx, mode_idx])**2
        
        # Interference from other modes
        interference_power = 0
        for i in range(self.num_modes):
            if i != mode_idx:
                interference_power += self.tx_power_W * np.abs(self.H[mode_idx, i])**2
        
        # Enhanced thermal noise power calculation
        noise_power = self._calculate_enhanced_noise_power()
        
        # Calculate SINR
        # Add a small epsilon to avoid division by zero
        denominator = interference_power + noise_power
        denominator = max(denominator, 1e-15)
        sinr = signal_power / denominator
        
        # Handle potential NaN or infinity
        if np.isnan(sinr) or np.isinf(sinr):
            sinr = 1e-15
        
        # Convert to dB with safety check
        if sinr > 0:
            sinr_dB = 10 * np.log10(sinr)
        else:
            sinr_dB = -150.0  # Very low SINR for zero signal
        
        # Validate output - allow wider realistic range
        sinr_dB = max(min(sinr_dB, 60.0), -150.0)  # Remove -40 dB floor
        
        return self.H, sinr_dB
    
    def _calculate_enhanced_noise_power(self) -> float:
        """
        Calculate enhanced noise power including all relevant factors.
        
        Returns:
            Total noise power in Watts
        """
        # Boltzmann constant
        k_boltzmann = 1.38e-23  # J/K
        
        # Convert noise figure from dB to linear
        noise_figure_linear = 10 ** (self.noise_figure_dB / 10)
        
        # Basic thermal noise power: P_n = k * T * B * NF
        thermal_noise = k_boltzmann * float(self.noise_temp) * float(self.bandwidth) * noise_figure_linear
        
        # Minimal additional noise sources
        # Keep atmospheric and quantum noise very small
        atmospheric_noise = thermal_noise * 0.01  # 1% of thermal noise
        quantum_noise = thermal_noise * 0.001  # 0.1% of thermal noise  
        phase_noise = thermal_noise * 0.01  # 1% of thermal noise
        
        # Total noise power - dominated by thermal noise
        total_noise = thermal_noise + atmospheric_noise + quantum_noise + phase_noise
        
        # Apply only implementation losses (not antenna efficiency - that affects signal)
        implementation_loss_dB = getattr(self, 'implementation_loss_dB', 3.0)
        implementation_loss_linear = 10 ** (implementation_loss_dB / 10)
        total_noise = total_noise * implementation_loss_linear
        
        # Small safety margin for unknown noise sources
        safety_margin_dB = 0.5  # Minimal margin
        safety_factor = 10 ** (safety_margin_dB / 10)
        total_noise = total_noise * safety_factor
        
        return total_noise 