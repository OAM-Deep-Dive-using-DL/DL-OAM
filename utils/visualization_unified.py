"""
Unified Visualization Module for OAM 6G

This module consolidates all visualization functionality into a single,
consistent interface.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy.special import assoc_laguerre

# Import all functions from existing modules
from .visualization_consolidated import (
    MetricsLogger,
    plot_training_curves,
    create_interactive_dashboard,
    visualize_q_values,
    plot_comparison,
    plot_metrics_grid,
    plot_heatmap
)

from .oam_visualizer_consolidated import (
    generate_oam_mode,
    plot_oam_mode,
    plot_multiple_oam_modes,
    plot_oam_mode_3d,
    visualize_oam_propagation,
    generate_oam_superposition,
    plot_oam_superposition,
    apply_turbulence,
    apply_pointing_error,
    visualize_oam_impairments,
    visualize_oam_mode_coupling,
    visualize_oam_crosstalk_matrix,
    generate_all_visualizations
)

# Create alias for evaluation dashboard
create_evaluation_dashboard = create_interactive_dashboard

# Export all functions
__all__ = [
    # Metrics logging
    'MetricsLogger',
    
    # Training visualization
    'plot_training_curves',
    'create_interactive_dashboard',
    'visualize_q_values',
    'plot_comparison',
    'plot_metrics_grid',
    'plot_heatmap',
    
    # OAM visualization
    'generate_oam_mode',
    'plot_oam_mode',
    'plot_multiple_oam_modes',
    'plot_oam_mode_3d',
    'visualize_oam_propagation',
    'generate_oam_superposition',
    'plot_oam_superposition',
    'apply_turbulence',
    'apply_pointing_error',
    'visualize_oam_impairments',
    'visualize_oam_mode_coupling',
    'visualize_oam_crosstalk_matrix',
    'generate_all_visualizations',
    
    # Evaluation dashboard
    'create_evaluation_dashboard',
] 