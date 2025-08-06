# Performance Plots

This directory contains performance analysis plots and benchmarking results for the OAM 6G system.

## Files

### Simulator Performance
- **`simulator_performance_comparison.png`**: Comparison of original vs optimized simulator performance
- **`simulator_time_distribution.png`**: Breakdown of time spent by different simulator components

### Analysis Results
- **`three_way_relationship_analysis.png`**: Analysis of handover count vs distance vs throughput relationships
- **`throughput_handover_tradeoff_analysis.png`**: Analysis of throughput vs handover frequency tradeoffs

## Purpose

These plots provide:
- **Performance Metrics**: Quantified performance improvements and bottlenecks
- **Optimization Insights**: Areas for further system optimization
- **Benchmarking**: Performance comparison between different implementations
- **Analysis Results**: Visual representation of system behavior analysis

## Key Insights

### Simulator Performance
- **Speedup**: 1.79x improvement in optimized simulator
- **Component Breakdown**: Turbulence (26.7%) and Crosstalk (24.3%) are the most time-consuming
- **Optimization Focus**: Areas identified for further performance improvements

### System Analysis
- **Distance Impact**: Strong correlation between distance and throughput
- **Handover Effects**: Relationship between handover frequency and system performance
- **Three-way Tradeoffs**: Complex interactions between distance, handovers, and throughput 