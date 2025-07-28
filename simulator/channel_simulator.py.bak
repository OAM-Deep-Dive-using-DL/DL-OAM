import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import scipy.special as sp
from scipy.constants import c as speed_of_light


class ChannelSimulator:
    """
    Basic physics simulator for OAM wireless channels.
    
    Simulates fundamental physical impairments that affect OAM mode transmission:
    - Path loss
    - Crosstalk between OAM modes
    - Rician fading
    - Pointing errors
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
    
    def _calculate_crosstalk(self, distance: float) -> np.ndarray:
        """
        Calculate basic crosstalk between OAM modes.
        
        Args:
            distance: Propagation distance in meters
            
        Returns:
            Crosstalk matrix
        """
        # Initialize crosstalk matrix
        crosstalk_matrix = np.eye(self.num_modes, dtype=complex)
        
        # Add basic crosstalk between adjacent modes
        # Crosstalk increases with distance
        distance_factor = min(1.0, distance / 1000.0)
        
        for i in range(self.num_modes):
            mode_i = i + self.min_mode
            for j in range(self.num_modes):
                if i != j:
                    mode_j = j + self.min_mode
                    mode_diff = abs(mode_i - mode_j)
                    
                    # Simple crosstalk model - decreases with mode difference
                    if mode_diff == 1:
                        coupling = 0.1 * distance_factor  # 10% coupling for adjacent modes
                    elif mode_diff == 2:
                        coupling = 0.05 * distance_factor  # 5% coupling for modes 2 steps apart
                    else:
                        coupling = 0.02 * distance_factor / mode_diff  # Lower coupling for distant modes
                    
                    # Add random phase
                    phase = np.random.uniform(0, 2 * np.pi)
                    crosstalk_matrix[i, j] = coupling * np.exp(1j * phase)
        
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
        mode_sensitivity = 1.0 + 0.1 * (oam_mode - self.min_mode)
        
        # Calculate loss using Gaussian beam model
        pointing_loss = np.exp(-(pointing_error * mode_sensitivity)**2 / (2 * self.beam_width**2))
        
        return max(pointing_loss, 0.01)  # Minimum 1% transmission
    
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
        
        # 2. Crosstalk
        crosstalk_matrix = self._calculate_crosstalk(distance)
        
        # 3. Rician fading
        fading_matrix = self._get_rician_fading_gain()
        
        # 4. Pointing error (specific to current mode)
        pointing_loss = self._get_pointing_error_loss(current_oam_mode)
        
        # Combine all effects to get channel matrix H
        # Add a small epsilon to avoid division by zero
        path_loss = max(path_loss, 1e-10)
        
        # Calculate channel gain (inverse of losses)
        channel_gain = 1.0 / path_loss
        
        # Apply antenna efficiency to signal
        channel_gain = channel_gain * self.antenna_efficiency
        
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