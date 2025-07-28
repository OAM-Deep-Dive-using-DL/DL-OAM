import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import scipy.special as sp
from scipy.constants import c as speed_of_light


class ChannelSimulator:
    """
    Physics simulator for OAM wireless channels with realistic atmospheric effects.
    
    Simulates physical impairments that affect OAM mode transmission:
    - Path loss
    - Atmospheric turbulence (Kolmogorov model)
    - Crosstalk between OAM modes (physics-based)
    - Rician fading
    - Pointing errors (with OAM mode sensitivity)
    - Atmospheric attenuation (frequency dependent)
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
        self.noise_figure_dB = 8.0  # 8 dB
        self.noise_temp = 290.0  # K
        self.bandwidth = 400e6  # 400 MHz
        
        # OAM parameters
        self.min_mode = 1
        self.max_mode = 8
        self.mode_spacing = 1
        self.beam_width = 0.03  # 30 mrad
        
        # Environment parameters  
        self.pointing_error_std = 0.005  # 5 mrad
        self.rician_k_factor = 8.0  # 8 dB
        
        # Atmospheric parameters
        self.turbulence_strength = 1e-14  # Cn² value (typical clear air)
        self.humidity = 50.0  # Relative humidity (%)
        self.temperature = 20.0  # Temperature (°C)
        self.pressure = 101.3  # Atmospheric pressure (kPa)
        
        # Basic additional parameters
        self.antenna_efficiency = 0.75  # 75% efficiency
        self.implementation_loss_dB = 3.0  # 3 dB losses
        
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
            if 'pointing_error_std' in env_config:
                self.pointing_error_std = float(env_config['pointing_error_std'])
            if 'rician_k_factor' in env_config:
                self.rician_k_factor = float(env_config['rician_k_factor'])
            if 'turbulence_strength' in env_config:
                self.turbulence_strength = float(env_config['turbulence_strength'])
            if 'humidity' in env_config:
                self.humidity = float(env_config['humidity'])
            if 'temperature' in env_config:
                self.temperature = float(env_config['temperature'])
            if 'pressure' in env_config:
                self.pressure = float(env_config['pressure'])
        
        # Update enhanced parameters
        if 'enhanced_params' in config:
            enhanced_config = config['enhanced_params']
            if 'antenna_efficiency' in enhanced_config:
                self.antenna_efficiency = float(enhanced_config['antenna_efficiency'])
            if 'implementation_loss_dB' in enhanced_config:
                self.implementation_loss_dB = float(enhanced_config['implementation_loss_dB'])
    
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
        
        # Pointing error validation
        if not (0.0001 <= self.pointing_error_std <= 0.1):  # 0.1 mrad to 100 mrad
            raise ValueError(f"Pointing error {self.pointing_error_std} rad is outside reasonable range")
        
        # Efficiency validation
        if not (0.1 <= self.antenna_efficiency <= 1.0):
            raise ValueError(f"Antenna efficiency {self.antenna_efficiency} is outside reasonable range (0.1-1.0)")
        
        # Turbulence validation
        if not (1e-17 <= self.turbulence_strength <= 1e-12):
            raise ValueError(f"Turbulence strength {self.turbulence_strength} is outside reasonable range (1e-17 to 1e-12)")
        
        # Atmospheric parameters validation
        if not (0 <= self.humidity <= 100):
            raise ValueError(f"Humidity {self.humidity}% is outside reasonable range (0-100%)")
        
        if not (-50 <= self.temperature <= 50):
            raise ValueError(f"Temperature {self.temperature}°C is outside reasonable range (-50 to 50°C)")
        
        if not (50 <= self.pressure <= 120):
            raise ValueError(f"Pressure {self.pressure} kPa is outside reasonable range (50-120 kPa)")
        
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
        Generate atmospheric turbulence effects using Kolmogorov model.
        
        Args:
            distance: Propagation distance in meters
            
        Returns:
            Turbulence-induced phase screen for each OAM mode
        """
        # Fried parameter (r0) - coherence length of turbulence
        # r0 = (0.423 * k^2 * Cn^2 * L)^(-3/5)
        r0 = (0.423 * (self.k ** 2) * self.turbulence_strength * distance) ** (-3/5)
        
        # Phase structure function
        # D_phi(r) = 6.88 * (r/r0)^(5/3)
        
        # Initialize phase perturbation for each mode
        phase_screen = np.zeros((self.num_modes, self.num_modes), dtype=complex)
        
        # Calculate beam radius at distance L
        w_L = self.beam_width * distance
        
        # Calculate scintillation index
        # For weak turbulence: σ_I^2 = 1.23 * Cn^2 * k^(7/6) * L^(11/6)
        scintillation_index = min(1.0, 1.23 * self.turbulence_strength * (self.k ** (7/6)) * (distance ** (11/6)))
        
        # Calculate beam wander
        # <r_c^2> = 2.42 * Cn^2 * L^3 * λ^(-1/3)
        beam_wander_variance = 2.42 * self.turbulence_strength * (distance ** 3) * (self.wavelength ** (-1/3))
        beam_wander = np.sqrt(beam_wander_variance)
        
        # Apply turbulence effects to each mode
        for i in range(self.num_modes):
            mode_i = i + self.min_mode
            
            # Higher OAM modes are more affected by turbulence
            mode_factor = (mode_i ** 2) / 4.0
            
            # Phase perturbation
            phase_variance = mode_factor * (w_L / r0) ** (5/3)
            phase_perturbation = np.random.normal(0, np.sqrt(phase_variance))
            
            # Amplitude fluctuation due to scintillation
            amplitude_factor = 1.0 - mode_factor * scintillation_index/2.0
            amplitude_factor = max(0.1, amplitude_factor)  # Limit the attenuation
            
            # Combined effect
            phase_screen[i, i] = amplitude_factor * np.exp(1j * phase_perturbation)
            
            # Off-diagonal elements (mode coupling due to turbulence)
            for j in range(self.num_modes):
                if i != j:
                mode_j = j + self.min_mode
                    mode_diff = abs(mode_i - mode_j)
                    
                    # Mode coupling strength decreases with mode difference
                    # and increases with turbulence strength
                    coupling_strength = (beam_wander / w_L) * (1.0 / mode_diff) * (w_L / r0) ** (5/3)
                    coupling_strength = min(0.3, coupling_strength)  # Limit the coupling
                    
                    # Random phase for coupling
                    coupling_phase = np.random.uniform(0, 2 * np.pi)
                    phase_screen[i, j] = coupling_strength * np.exp(1j * coupling_phase)
        
        return phase_screen
    
    def _calculate_crosstalk(self, distance: float, turbulence_screen: np.ndarray) -> np.ndarray:
        """
        Calculate crosstalk between OAM modes with physics-based model.
        
        Args:
            distance: Propagation distance in meters
            turbulence_screen: Turbulence-induced phase screen
            
        Returns:
            Crosstalk matrix
        """
        # Initialize crosstalk matrix
        crosstalk_matrix = np.eye(self.num_modes, dtype=complex)
        
        # Diffraction-induced crosstalk
        # Increases with distance and decreases with mode spacing
        diffraction_factor = self.wavelength * distance / (np.pi * self.beam_width**2)
        
        for i in range(self.num_modes):
            mode_i = i + self.min_mode
            for j in range(self.num_modes):
                if i != j:
                mode_j = j + self.min_mode
                mode_diff = abs(mode_i - mode_j)
                
                    # Diffraction-based coupling
                    # Higher modes and larger differences have lower coupling
                    diffraction_coupling = diffraction_factor / (mode_i * mode_diff**2)
                    diffraction_coupling = min(0.15, diffraction_coupling)  # Limit max coupling
                    
                    # Add turbulence-induced coupling
                    total_coupling = diffraction_coupling + abs(turbulence_screen[i, j])
                    
                    # Random phase for total coupling
                    coupling_phase = np.random.uniform(0, 2 * np.pi)
                    crosstalk_matrix[i, j] = total_coupling * np.exp(1j * coupling_phase)
        
        # Normalize to ensure energy conservation
        for i in range(self.num_modes):
            # Calculate total power in row
            total_power = np.sum(np.abs(crosstalk_matrix[i, :])**2)
            if total_power > 1e-15:
                # Normalize but preserve relative coupling strengths
                crosstalk_matrix[i, :] = crosstalk_matrix[i, :] / np.sqrt(total_power)
        
        return crosstalk_matrix
        
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
        Calculate loss due to pointing errors with OAM mode sensitivity.
        
        Pointing errors occur when there's misalignment between the transmitter and receiver.
        Higher OAM modes are more sensitive to pointing errors due to their more complex
        phase structure.
        
        Args:
            oam_mode: Current OAM mode
            
        Returns:
            Pointing error loss in linear scale
        """
        # Generate random pointing error
        pointing_error = np.random.normal(0, self.pointing_error_std)
        
        # Higher OAM modes are more sensitive to pointing errors
        # Based on theoretical model: sensitivity ~ |l|
        mode_sensitivity = 1.0 + 0.2 * (oam_mode - self.min_mode)
        
        # Calculate loss using Gaussian beam model with mode-dependent sensitivity
        pointing_loss = np.exp(-(pointing_error * mode_sensitivity)**2 / (2 * self.beam_width**2))
        
        return max(pointing_loss, 0.01)  # Minimum 1% transmission
    
    def _get_atmospheric_attenuation(self, distance: float) -> float:
        """
        Calculate atmospheric attenuation based on frequency, humidity, temperature, and pressure.
        
        Args:
            distance: Propagation distance in meters
            
        Returns:
            Atmospheric attenuation in linear scale
        """
        # Convert frequency to GHz for attenuation calculations
        freq_GHz = self.frequency / 1e9
        
        # Base attenuation calculation
        # Simplified ITU-R model for atmospheric gases
        if freq_GHz < 60:  # Below 60 GHz
            # Oxygen and water vapor attenuation (simplified model)
            gamma_oxygen = 0.001 * (freq_GHz**1.5) * (self.pressure/101.3)
            gamma_water = 0.0001 * freq_GHz * self.humidity * np.exp(self.temperature/15)
            
            # Total specific attenuation (dB/km)
            gamma_total = gamma_oxygen + gamma_water
        else:  # Above 60 GHz (stronger absorption)
            # Higher attenuation due to molecular resonances
            gamma_oxygen = 0.01 * freq_GHz * (self.pressure/101.3)
            gamma_water = 0.001 * (freq_GHz**1.5) * self.humidity * np.exp(self.temperature/20)
            
            # Total specific attenuation (dB/km)
            gamma_total = gamma_oxygen + gamma_water
        
        # Rain attenuation is not included in this model
        # Could be added based on ITU-R P.838 if needed
        
        # Convert from dB/km to linear scale for the given distance
        distance_km = distance / 1000.0
        attenuation_dB = gamma_total * distance_km
        attenuation_linear = 10 ** (attenuation_dB / 10)
        
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
        
        # 3. Crosstalk with turbulence effects
        crosstalk_matrix = self._calculate_crosstalk(distance, turbulence_screen)
        
        # 4. Rician fading
        fading_matrix = self._get_rician_fading_gain()
        
        # 5. Pointing error (specific to current mode)
        pointing_loss = self._get_pointing_error_loss(current_oam_mode)
        
        # 6. Atmospheric attenuation
        atmospheric_attenuation = self._get_atmospheric_attenuation(distance)
        
        # Combine all effects to get channel matrix H
        # Add a small epsilon to avoid division by zero
        path_loss = max(path_loss, 1e-10)
        
        # Calculate channel gain (inverse of losses)
        channel_gain = 1.0 / (path_loss * atmospheric_attenuation)
        
        # Apply antenna efficiency to signal
        channel_gain = channel_gain * self.antenna_efficiency
        
        # Start with crosstalk matrix and apply channel gain
        self.H = crosstalk_matrix * fading_matrix * turbulence_screen * np.sqrt(channel_gain)
        
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
        
        # Thermal noise power calculation
        noise_power = self._calculate_noise_power()
        
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
        
        # Validate output
        sinr_dB = max(min(sinr_dB, 60.0), -40.0)
        
        return self.H, sinr_dB
    
    def _calculate_noise_power(self) -> float:
        """
        Calculate basic noise power.
        
        Returns:
            Total noise power in Watts
        """
        # Boltzmann constant
        k_boltzmann = 1.38e-23  # J/K
        
        # Convert noise figure from dB to linear
        noise_figure_linear = 10 ** (self.noise_figure_dB / 10)
        
        # Basic thermal noise power: P_n = k * T * B * NF
        thermal_noise = k_boltzmann * self.noise_temp * self.bandwidth * noise_figure_linear
        
        # Apply implementation losses
        implementation_loss_linear = 10 ** (self.implementation_loss_dB / 10)
        total_noise = thermal_noise * implementation_loss_linear
        
        return total_noise 
