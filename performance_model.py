"""
Unified Performance Model Module for THz LEO-ISL ISAC Network Simulation
=========================================================================

This module serves as the central hub connecting all physical models to final
performance metrics. It implements the unified SINR and measurement variance
formulas that consistently appear throughout the theoretical framework.

Based on:
- Section III equations (17)-(18) for measurement variance and effective SINR
- Noise-normalized unified approach (Plan B from expert review)
- Hardware impairment model from Section 2.3
- Network interference framework from Section 4.2

Author: THz ISAC Research Team
Date: August 2025
"""

import numpy as np
from typing import Union, List, Optional, Dict, Tuple
from dataclasses import dataclass
import warnings

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s


@dataclass
class PerformanceMetrics:
    """
    Container for comprehensive performance metrics.
    
    Attributes:
        sinr_eff: Effective SINR (linear scale)
        sinr_eff_db: Effective SINR in dB
        measurement_variance: TOA measurement error variance (s²)
        ranging_rmse: Ranging RMSE (meters)
        hardware_penalty_db: Performance loss due to hardware (dB)
        interference_penalty_db: Performance loss due to interference (dB)
    """
    sinr_eff: float
    sinr_eff_db: float
    measurement_variance: float
    ranging_rmse: float
    hardware_penalty_db: float
    interference_penalty_db: float


def calculate_effective_sinr(snr0: float,
                            gamma_eff: float = 0.0,
                            sigma_phi_squared: float = 0.0,
                            normalized_interference: Union[float, List[float], np.ndarray] = 0.0,
                            hardware_on: bool = True,
                            interference_on: bool = True,
                            phase_noise_on: bool = True) -> float:
    """
    Calculate effective SINR incorporating all impairments.
    
    Implements the unified noise-normalized formula from Section III eq. (18):
    SINR_eff,ℓ = (SNR_0,ℓ · exp(-σ²_φ,ℓ)) / (1 + SNR_0,ℓ · Γ_eff,ℓ + Σ α̃_ℓm)
    
    This is the "Plan B" approach from expert review where all terms in the
    denominator are normalized by noise power, making them dimensionless.
    
    Args:
        snr0: Pre-impairment SNR (linear scale, not dB)
        gamma_eff: Effective hardware quality factor (default 0 = ideal)
        sigma_phi_squared: Phase noise variance in rad² (default 0 = no phase noise)
        normalized_interference: Normalized interference coefficients α̃_ℓm
                                Can be scalar, list, or numpy array
        hardware_on: Enable hardware impairments (default True)
        interference_on: Enable interference effects (default True)
        phase_noise_on: Enable phase noise effects (default True)
    
    Returns:
        Effective SINR (linear scale, not dB)
    
    Example:
        >>> # 20 dB SNR with moderate hardware and interference
        >>> snr0 = 100
        >>> gamma_eff = 0.01  # High-performance hardware
        >>> sigma_phi_sq = 1e-4  # Phase noise
        >>> interference = [0.5, 0.3, 0.1]  # Three interferers
        >>> sinr_eff = calculate_effective_sinr(
        ...     snr0, gamma_eff, sigma_phi_sq, interference
        ... )
    """
    # Input validation
    if snr0 < 0:
        raise ValueError("SNR must be non-negative")
    if gamma_eff < 0 or gamma_eff > 1:
        warnings.warn("Hardware quality factor should be in [0, 1]")
    if sigma_phi_squared < 0:
        raise ValueError("Phase noise variance must be non-negative")
    
    # Apply configuration switches
    if not hardware_on:
        gamma_eff = 0.0
    if not phase_noise_on:
        sigma_phi_squared = 0.0
    
    # Calculate coherent power loss due to phase noise
    coherent_loss = np.exp(-sigma_phi_squared) if phase_noise_on else 1.0
    
    # Calculate total normalized interference
    if interference_on and normalized_interference is not None:
        if isinstance(normalized_interference, (list, np.ndarray)):
            interference_sum = np.sum(normalized_interference)
        else:
            interference_sum = float(normalized_interference)
    else:
        interference_sum = 0.0
    
    if interference_sum < 0:
        raise ValueError("Interference sum must be non-negative")
    
    # Calculate effective SINR using unified formula
    numerator = snr0 * coherent_loss
    denominator = 1 + snr0 * gamma_eff + interference_sum
    
    sinr_eff = numerator / denominator
    
    return sinr_eff


def calculate_measurement_variance(sinr_eff: float,
                                  sigma_phi_squared: float,
                                  f_c: float,
                                  kappa_wf: Optional[float] = None,
                                  bandwidth: Optional[float] = None) -> float:
    """
    Calculate TOA measurement error variance.
    
    Implements equation (17) from Section III:
    σ²_meas,ℓ = c² [κ_WF/SINR_eff,ℓ + σ²_φ,ℓ/(2πf_c)²]
    
    This unified model captures both:
    - Waveform-limited regime (first term dominates at low SINR)
    - Phase-noise-limited regime (second term provides error floor)
    
    Args:
        sinr_eff: Effective SINR (linear scale)
        sigma_phi_squared: Phase noise variance (rad²)
        f_c: Carrier frequency (Hz)
        kappa_wf: Waveform constant κ_WF = 1/(8π²β²)
                 If None, calculated from bandwidth
        bandwidth: Signal bandwidth (Hz), used if kappa_wf not provided
    
    Returns:
        TOA measurement error variance (s²)
    
    Example:
        >>> sinr_eff = 50  # ~17 dB
        >>> sigma_phi_sq = 1e-4
        >>> f_c = 300e9  # 300 GHz
        >>> bandwidth = 10e9  # 10 GHz
        >>> var = calculate_measurement_variance(
        ...     sinr_eff, sigma_phi_sq, f_c, bandwidth=bandwidth
        ... )
    """
    # Input validation
    if sinr_eff <= 0:
        raise ValueError("Effective SINR must be positive")
    if sigma_phi_squared < 0:
        raise ValueError("Phase noise variance must be non-negative")
    if f_c <= 0:
        raise ValueError("Carrier frequency must be positive")
    
    # Determine waveform constant
    if kappa_wf is None:
        if bandwidth is None:
            raise ValueError("Either kappa_wf or bandwidth must be provided")
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive")
        # κ_WF = 1/(8π²β²) where β is RMS bandwidth
        # For rectangular spectrum, β ≈ bandwidth/√12
        beta_rms = bandwidth / np.sqrt(12)
        kappa_wf = 1 / (8 * np.pi**2 * beta_rms**2)
    elif kappa_wf <= 0:
        raise ValueError("Waveform constant must be positive")
    
    # Waveform-limited term (CRLB for TOA estimation)
    waveform_term = kappa_wf / sinr_eff
    
    # Phase-noise-limited term (error floor)
    if sigma_phi_squared > 0:
        phase_noise_term = sigma_phi_squared / (2 * np.pi * f_c)**2
    else:
        phase_noise_term = 0.0
    
    # Total measurement variance
    sigma_squared_meas = SPEED_OF_LIGHT**2 * (waveform_term + phase_noise_term)
    
    return sigma_squared_meas


def calculate_ranging_rmse(measurement_variance: float) -> float:
    """
    Convert TOA measurement variance to ranging RMSE.
    
    Args:
        measurement_variance: TOA measurement error variance (s²)
    
    Returns:
        Ranging RMSE in meters
    """
    if measurement_variance < 0:
        raise ValueError("Measurement variance must be non-negative")
    
    # RMSE = c * sqrt(σ²_τ) where σ²_τ is timing variance
    timing_std = np.sqrt(measurement_variance)
    ranging_rmse = SPEED_OF_LIGHT * timing_std
    
    return ranging_rmse


def analyze_performance_breakdown(snr0: float,
                                 gamma_eff: float,
                                 sigma_phi_squared: float,
                                 normalized_interference: Union[float, List[float]],
                                 f_c: float,
                                 bandwidth: float) -> PerformanceMetrics:
    """
    Comprehensive performance analysis with breakdown by impairment source.
    
    This function calculates all performance metrics and identifies the
    dominant impairment sources, useful for system optimization.
    
    Args:
        snr0: Pre-impairment SNR (linear scale)
        gamma_eff: Hardware quality factor
        sigma_phi_squared: Phase noise variance (rad²)
        normalized_interference: Normalized interference coefficients
        f_c: Carrier frequency (Hz)
        bandwidth: Signal bandwidth (Hz)
    
    Returns:
        PerformanceMetrics object with comprehensive analysis
    
    Example:
        >>> metrics = analyze_performance_breakdown(
        ...     snr0=100, gamma_eff=0.01, sigma_phi_squared=1e-4,
        ...     normalized_interference=[0.5, 0.3], f_c=300e9, bandwidth=10e9
        ... )
        >>> print(f"Effective SINR: {metrics.sinr_eff_db:.1f} dB")
        >>> print(f"Ranging RMSE: {metrics.ranging_rmse*1e3:.2f} mm")
    """
    # Calculate SINR with all impairments
    sinr_full = calculate_effective_sinr(
        snr0, gamma_eff, sigma_phi_squared, normalized_interference,
        hardware_on=True, interference_on=True, phase_noise_on=True
    )
    
    # Calculate SINR without hardware impairments (for penalty calculation)
    sinr_no_hw = calculate_effective_sinr(
        snr0, 0, sigma_phi_squared, normalized_interference,
        hardware_on=False, interference_on=True, phase_noise_on=True
    )
    
    # Calculate SINR without interference (for penalty calculation)
    sinr_no_int = calculate_effective_sinr(
        snr0, gamma_eff, sigma_phi_squared, 0,
        hardware_on=True, interference_on=False, phase_noise_on=True
    )
    
    # Calculate measurement variance and ranging RMSE
    meas_var = calculate_measurement_variance(
        sinr_full, sigma_phi_squared, f_c, bandwidth=bandwidth
    )
    ranging_rmse = calculate_ranging_rmse(meas_var)
    
    # Calculate penalties in dB
    hardware_penalty = 10 * np.log10(sinr_no_hw / sinr_full) if sinr_full > 0 else np.inf
    interference_penalty = 10 * np.log10(sinr_no_int / sinr_full) if sinr_full > 0 else np.inf
    
    return PerformanceMetrics(
        sinr_eff=sinr_full,
        sinr_eff_db=10 * np.log10(sinr_full) if sinr_full > 0 else -np.inf,
        measurement_variance=meas_var,
        ranging_rmse=ranging_rmse,
        hardware_penalty_db=hardware_penalty,
        interference_penalty_db=interference_penalty
    )


def identify_limiting_regime(snr0: float,
                            gamma_eff: float,
                            normalized_interference_sum: float) -> str:
    """
    Identify the dominant performance-limiting factor.
    
    Analyzes the denominator terms in the SINR formula to determine
    which factor is the primary bottleneck.
    
    Args:
        snr0: Pre-impairment SNR (linear scale)
        gamma_eff: Hardware quality factor
        normalized_interference_sum: Sum of normalized interference
    
    Returns:
        String describing the limiting regime
    """
    # Calculate denominator terms
    noise_term = 1.0
    hardware_term = snr0 * gamma_eff
    interference_term = normalized_interference_sum
    
    # Find dominant term
    terms = {
        "Noise-limited": noise_term,
        "Hardware-limited": hardware_term,
        "Interference-limited": interference_term
    }
    
    dominant = max(terms, key=terms.get)
    
    # Add quantitative assessment
    total = noise_term + hardware_term + interference_term
    dominance_percentage = 100 * terms[dominant] / total
    
    return f"{dominant} ({dominance_percentage:.1f}% of degradation)"


def calculate_capacity_upper_bound(sinr_eff: float) -> float:
    """
    Calculate Shannon capacity upper bound.
    
    C = log₂(1 + SINR_eff) bits/symbol
    
    Args:
        sinr_eff: Effective SINR (linear scale)
    
    Returns:
        Capacity in bits/symbol
    """
    if sinr_eff < 0:
        raise ValueError("SINR must be non-negative")
    
    capacity = np.log2(1 + sinr_eff)
    return capacity


def calculate_hardware_ceiling(gamma_eff: float, 
                              sigma_phi_squared: float) -> Dict[str, float]:
    """
    Calculate hardware-imposed performance ceilings.
    
    As SNR → ∞, performance saturates at hardware-determined limits.
    
    Args:
        gamma_eff: Hardware quality factor
        sigma_phi_squared: Phase noise variance (rad²)
    
    Returns:
        Dictionary with ceiling values for SINR and capacity
    """
    if gamma_eff <= 0:
        return {"sinr_ceiling": np.inf, "capacity_ceiling": np.inf}
    
    # SINR ceiling from Section 2.3
    sinr_ceiling = np.exp(-sigma_phi_squared) / gamma_eff
    
    # Capacity ceiling
    capacity_ceiling = np.log2(1 + sinr_ceiling)
    
    return {
        "sinr_ceiling": sinr_ceiling,
        "sinr_ceiling_db": 10 * np.log10(sinr_ceiling),
        "capacity_ceiling": capacity_ceiling
    }


# ============================================================================
# Unit Tests
# ============================================================================

def test_effective_sinr():
    """Test effective SINR calculation with various configurations."""
    print("Testing effective SINR calculation...")
    
    # Test parameters
    snr0 = 100  # 20 dB
    gamma_eff = 0.01
    sigma_phi_sq = 1e-4
    interference = [0.5, 0.3, 0.1]
    
    # Test with all impairments
    sinr_full = calculate_effective_sinr(
        snr0, gamma_eff, sigma_phi_sq, interference
    )
    print(f"  Full impairments: {10*np.log10(sinr_full):.2f} dB")
    
    # Test with hardware off
    sinr_no_hw = calculate_effective_sinr(
        snr0, gamma_eff, sigma_phi_sq, interference, hardware_on=False
    )
    print(f"  Hardware off: {10*np.log10(sinr_no_hw):.2f} dB")
    
    # Test with interference off
    sinr_no_int = calculate_effective_sinr(
        snr0, gamma_eff, sigma_phi_sq, interference, interference_on=False
    )
    print(f"  Interference off: {10*np.log10(sinr_no_int):.2f} dB")
    
    # Test with phase noise off
    sinr_no_pn = calculate_effective_sinr(
        snr0, gamma_eff, sigma_phi_sq, interference, phase_noise_on=False
    )
    print(f"  Phase noise off: {10*np.log10(sinr_no_pn):.2f} dB")
    
    # Verify hierarchy
    assert sinr_full < sinr_no_hw, "Hardware should reduce SINR"
    assert sinr_full < sinr_no_int, "Interference should reduce SINR"
    assert sinr_full < sinr_no_pn, "Phase noise should reduce SINR"
    
    print("✓ Effective SINR tests passed")


def test_measurement_variance():
    """Test measurement variance calculation."""
    print("Testing measurement variance calculation...")
    
    # Test parameters
    sinr_eff = 50  # ~17 dB
    sigma_phi_sq = 1e-4
    f_c = 300e9  # 300 GHz
    bandwidth = 10e9  # 10 GHz
    
    # Calculate variance
    var = calculate_measurement_variance(
        sinr_eff, sigma_phi_sq, f_c, bandwidth=bandwidth
    )
    
    # Convert to ranging RMSE
    rmse = calculate_ranging_rmse(var)
    
    print(f"  Measurement variance: {var:.3e} s²")
    print(f"  Ranging RMSE: {rmse*1e3:.3f} mm")
    
    # Test error floor
    var_high_sinr = calculate_measurement_variance(
        1e6, sigma_phi_sq, f_c, bandwidth=bandwidth
    )
    rmse_floor = calculate_ranging_rmse(var_high_sinr)
    
    print(f"  Error floor (high SINR): {rmse_floor*1e3:.3f} mm")
    
    assert rmse > rmse_floor, "Should approach but not exceed floor"
    
    print("✓ Measurement variance tests passed")


def test_performance_breakdown():
    """Test comprehensive performance analysis."""
    print("Testing performance breakdown...")
    
    # System parameters
    metrics = analyze_performance_breakdown(
        snr0=100,  # 20 dB
        gamma_eff=0.01,
        sigma_phi_squared=1e-4,
        normalized_interference=[0.5, 0.3, 0.1],
        f_c=300e9,
        bandwidth=10e9
    )
    
    print(f"  Effective SINR: {metrics.sinr_eff_db:.1f} dB")
    print(f"  Ranging RMSE: {metrics.ranging_rmse*1e3:.2f} mm")
    print(f"  Hardware penalty: {metrics.hardware_penalty_db:.1f} dB")
    print(f"  Interference penalty: {metrics.interference_penalty_db:.1f} dB")
    
    assert metrics.hardware_penalty_db > 0, "Should have hardware penalty"
    assert metrics.interference_penalty_db > 0, "Should have interference penalty"
    
    print("✓ Performance breakdown tests passed")


def test_limiting_regime():
    """Test regime identification."""
    print("Testing limiting regime identification...")
    
    # Noise-limited case
    regime1 = identify_limiting_regime(snr0=1, gamma_eff=0.01, 
                                      normalized_interference_sum=0.1)
    print(f"  Low SNR: {regime1}")
    assert "Noise" in regime1
    
    # Hardware-limited case
    regime2 = identify_limiting_regime(snr0=1000, gamma_eff=0.05,
                                      normalized_interference_sum=0.1)
    print(f"  High SNR, poor hardware: {regime2}")
    assert "Hardware" in regime2
    
    # Interference-limited case
    regime3 = identify_limiting_regime(snr0=100, gamma_eff=0.001,
                                      normalized_interference_sum=50)
    print(f"  Strong interference: {regime3}")
    assert "Interference" in regime3
    
    print("✓ Regime identification tests passed")


def test_hardware_ceiling():
    """Test hardware-imposed performance ceilings."""
    print("Testing hardware ceiling calculation...")
    
    # Calculate ceilings for different hardware qualities
    for gamma, name in [(0.005, "State-of-the-Art"), 
                        (0.01, "High-Performance"),
                        (0.05, "Low-Cost")]:
        ceiling = calculate_hardware_ceiling(gamma, 1e-4)
        print(f"  {name}: SINR ceiling = {ceiling['sinr_ceiling_db']:.1f} dB, "
              f"Capacity = {ceiling['capacity_ceiling']:.2f} bits/symbol")
    
    # Verify ordering
    ceiling_good = calculate_hardware_ceiling(0.01, 1e-4)
    ceiling_bad = calculate_hardware_ceiling(0.05, 1e-4)
    
    assert ceiling_good["sinr_ceiling"] > ceiling_bad["sinr_ceiling"], \
        "Better hardware should have higher ceiling"
    
    print("✓ Hardware ceiling tests passed")


def test_unity_switches():
    """Test that switches properly disable effects."""
    print("Testing configuration switches...")
    
    snr0 = 100
    gamma = 0.05
    sigma_phi = 1e-3
    interference = 10.0
    
    # All effects on (baseline)
    sinr_all = calculate_effective_sinr(snr0, gamma, sigma_phi, interference)
    
    # All effects off (should equal SNR0)
    sinr_none = calculate_effective_sinr(
        snr0, gamma, sigma_phi, interference,
        hardware_on=False, interference_on=False, phase_noise_on=False
    )
    
    print(f"  All effects: {10*np.log10(sinr_all):.2f} dB")
    print(f"  No effects: {10*np.log10(sinr_none):.2f} dB")
    print(f"  Original SNR: {10*np.log10(snr0):.2f} dB")
    
    assert abs(sinr_none - snr0) < 1e-10, "Should equal original SNR with all off"
    
    print("✓ Configuration switch tests passed")


if __name__ == "__main__":
    """Run all unit tests and demonstrate usage."""
    print("=" * 60)
    print("Running Performance Model Unit Tests")
    print("=" * 60)
    
    test_effective_sinr()
    test_measurement_variance()
    test_performance_breakdown()
    test_limiting_regime()
    test_hardware_ceiling()
    test_unity_switches()
    
    print("=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
    
    # Comprehensive example
    print("\n" + "=" * 60)
    print("Example: Complete Performance Analysis")
    print("=" * 60)
    
    # System configuration
    print("\nSystem Configuration:")
    print("-" * 40)
    snr0_db = 20
    snr0 = 10**(snr0_db/10)
    gamma_eff = 0.01  # High-performance hardware
    sigma_phi_squared = 1e-4  # Moderate phase noise
    f_c = 300e9  # 300 GHz
    bandwidth = 10e9  # 10 GHz
    
    print(f"Pre-impairment SNR: {snr0_db} dB")
    print(f"Hardware quality factor: {gamma_eff}")
    print(f"Phase noise variance: {sigma_phi_squared} rad²")
    print(f"Carrier frequency: {f_c/1e9:.0f} GHz")
    print(f"Bandwidth: {bandwidth/1e9:.0f} GHz")
    
    # Interference scenario
    print("\nInterference Scenario:")
    print("-" * 40)
    # Three interferers with different strengths
    alpha_values = [0.01, 0.005, 0.002]  # Interference coefficients
    normalized_interference = [snr0 * alpha for alpha in alpha_values]
    
    for i, (alpha, alpha_tilde) in enumerate(zip(alpha_values, normalized_interference)):
        print(f"Interferer {i+1}: α = {10*np.log10(alpha):.1f} dB, "
              f"α̃ = {alpha_tilde:.3f}")
    
    # Performance analysis
    print("\nPerformance Analysis:")
    print("-" * 40)
    
    metrics = analyze_performance_breakdown(
        snr0, gamma_eff, sigma_phi_squared,
        normalized_interference, f_c, bandwidth
    )
    
    print(f"Effective SINR: {metrics.sinr_eff_db:.2f} dB")
    print(f"Capacity: {calculate_capacity_upper_bound(metrics.sinr_eff):.2f} bits/symbol")
    print(f"Ranging RMSE: {metrics.ranging_rmse*1e3:.2f} mm")
    print(f"Hardware penalty: {metrics.hardware_penalty_db:.2f} dB")
    print(f"Interference penalty: {metrics.interference_penalty_db:.2f} dB")
    
    # Limiting regime
    regime = identify_limiting_regime(snr0, gamma_eff, sum(normalized_interference))
    print(f"Limiting regime: {regime}")
    
    # Hardware ceiling
    print("\nHardware-Imposed Ceilings:")
    print("-" * 40)
    ceiling = calculate_hardware_ceiling(gamma_eff, sigma_phi_squared)
    print(f"SINR ceiling: {ceiling['sinr_ceiling_db']:.1f} dB")
    print(f"Capacity ceiling: {ceiling['capacity_ceiling']:.2f} bits/symbol")
    
    # Comparison with ideal case
    print("\nComparison with Ideal System:")
    print("-" * 40)
    sinr_ideal = calculate_effective_sinr(
        snr0, 0, 0, 0,
        hardware_on=False, interference_on=False, phase_noise_on=False
    )
    capacity_ideal = calculate_capacity_upper_bound(sinr_ideal)
    
    print(f"Ideal SINR: {10*np.log10(sinr_ideal):.2f} dB")
    print(f"Ideal capacity: {capacity_ideal:.2f} bits/symbol")
    print(f"Total degradation: {10*np.log10(sinr_ideal/metrics.sinr_eff):.2f} dB")
    print(f"Capacity loss: {capacity_ideal - calculate_capacity_upper_bound(metrics.sinr_eff):.2f} bits/symbol")