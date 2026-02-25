"""
Cholera Risk Assessment - Simulation Model

This module contains the core Monte Carlo simulation logic for evaluating 
transboundary cholera infection risk. The model multiplies six probability 
components representing steps in the disease transmission chain:

    P_final = P1 × P2 × P3 × P4 × P5 × P6

Where:
    P1  = Probability of Cholera outbreak in source region (Poisson process)
    P2  = Probability of Cholera infection in humans (Beta distribution)
    P3  = False Negative clinical examination in source region (1 - Sensitivity)
    P4  = False Negative clinical examination in target region (1 - Sensitivity)
    P5  = Exposure risk due to unsafe water and sanitation access
    P6  = Probability of mortality (case fatality rate)

All parameters are configurable. Users can override any P value (P1–P6)
to adapt the model to any region or epidemiological scenario.
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from typing import Union


@dataclass
class SimulationConfig:
    """Configuration parameters for the cholera risk simulation.
    
    Attributes:
        n_samples: Number of Monte Carlo samples to generate.
        seed: Random seed for reproducibility.
        t: Time period (years) for the Poisson process.
        total_outbreaks: Total cholera outbreaks observed.
        observation_years: Number of years of observation data.
        observation_months_per_year: Months per year in the observation window.
        beta_alpha: Alpha parameter for the Beta distribution (P2).
        beta_beta: Beta parameter for the Beta distribution (P2).
        sensitivity_bangladesh_low: Lower bound of CholKit RDT sensitivity (Bangladesh).
        sensitivity_bangladesh_high: Upper bound of CholKit RDT sensitivity (Bangladesh).
        sensitivity_india_low: Lower bound of CholKit RDT sensitivity (India).
        sensitivity_india_high: Upper bound of CholKit RDT sensitivity (India).
        unsafe_water_fraction: Fraction of population with unsafe water access.
        unsafe_sanitation_fraction: Fraction of population with unsafe sanitation.
        case_fatality_rate: Cholera case fatality rate.
        P1_override: If set (float), use this fixed value for P1 instead of computing from Poisson.
        P2_override: If set (float), use this fixed value for P2 instead of sampling from Beta.
        P3_override: If set (float), use this fixed value for P3 instead of sampling from Uniform.
        P4_override: If set (float), use this fixed value for P4 instead of sampling from Uniform.
        P5_override: If set (float), use this fixed value for P5 instead of computing from config.
        P6_override: If set (float), use this fixed value for P6 instead of using case_fatality_rate.
    """
    n_samples: int = 10000
    seed: int = 42
    t: float = 1.0
    total_outbreaks: int = 2779
    observation_years: int = 22
    observation_months_per_year: int = 12
    beta_alpha: int = 1605
    beta_beta: int = 24618
    sensitivity_bangladesh_low: float = 0.549
    sensitivity_bangladesh_high: float = 0.906
    sensitivity_india_low: float = 0.884
    sensitivity_india_high: float = 0.999
    unsafe_water_fraction: float = 0.11
    unsafe_sanitation_fraction: float = 0.01
    case_fatality_rate: float = 0.03

    # ─── Direct P-value overrides (None = use default distribution) ───
    P1_override: Optional[float] = None
    P2_override: Optional[float] = None
    P3_override: Optional[float] = None
    P4_override: Optional[float] = None
    P5_override: Optional[float] = None
    P6_override: Optional[float] = None

    @property
    def lambda_rate(self) -> float:
        """Compute the Poisson rate parameter λ from observed data."""
        return self.total_outbreaks / (self.observation_years * self.observation_months_per_year)

    @property
    def P5(self) -> float:
        """Compute exposure risk from water and sanitation."""
        if self.P5_override is not None:
            return self.P5_override
        return self.unsafe_water_fraction * self.unsafe_sanitation_fraction

    @property
    def P6(self) -> float:
        """Return case fatality rate."""
        if self.P6_override is not None:
            return self.P6_override
        return self.case_fatality_rate


class CholeraSimulator:
    """Monte Carlo simulator for transboundary cholera risk assessment.
    
    This class encapsulates the entire simulation pipeline:
    1. Generate random samples from appropriate distributions
    2. Compute the compound probability of risk
    3. Calculate summary statistics  
    4. Perform sensitivity analysis using Spearman rank correlation
    
    Users can override any P value (P1–P6) to use a fixed value instead
    of the default probability distribution.
    
    Example:
        >>> # Default parameters
        >>> sim = CholeraSimulator(n_samples=10000, seed=42)
        >>> results = sim.run_simulation()
        
        >>> # With custom P values
        >>> sim = CholeraSimulator(n_samples=10000, P1=0.95, P5=0.002, P6=0.05)
        >>> results = sim.run_simulation()
        
        >>> # Override all P values with fixed numbers
        >>> sim = CholeraSimulator(P1=0.99, P2=0.06, P3=0.27, P4=0.06, P5=0.001, P6=0.03)
        >>> results = sim.run_simulation()
    """

    def __init__(self, n_samples: int = 10000, seed: int = 42,
                 config: Optional[SimulationConfig] = None,
                 P1: Optional[float] = None,
                 P2: Optional[float] = None,
                 P3: Optional[float] = None,
                 P4: Optional[float] = None,
                 P5: Optional[float] = None,
                 P6: Optional[float] = None):
        """Initialize the simulator.
        
        Args:
            n_samples: Number of Monte Carlo samples. Overrides config if provided.
            seed: Random seed for reproducibility. Overrides config if provided.
            config: Optional SimulationConfig with custom parameters.
                    If None, default epidemiological parameters are used.
            P1: Override P1 (outbreak probability). If None, computed from Poisson.
            P2: Override P2 (infection rate). If None, sampled from Beta distribution.
            P3: Override P3 (false negative Bangladesh). If None, sampled from Uniform.
            P4: Override P4 (false negative India). If None, sampled from Uniform.
            P5: Override P5 (exposure risk). If None, computed as water × sanitation.
            P6: Override P6 (mortality). If None, uses case_fatality_rate.
        """
        if config is not None:
            self.config = config
            self.config.n_samples = n_samples
            self.config.seed = seed
        else:
            self.config = SimulationConfig(n_samples=n_samples, seed=seed)

        # Apply P-value overrides from constructor args
        if P1 is not None:
            self.config.P1_override = P1
        if P2 is not None:
            self.config.P2_override = P2
        if P3 is not None:
            self.config.P3_override = P3
        if P4 is not None:
            self.config.P4_override = P4
        if P5 is not None:
            self.config.P5_override = P5
        if P6 is not None:
            self.config.P6_override = P6

        self._results: Optional[Dict[str, Any]] = None

    def run_simulation(self) -> Dict[str, Any]:
        """Execute the Monte Carlo simulation.
        
        Returns:
            Dictionary containing:
                - 'samples': Array of final risk probability samples (n_samples,)
                - 'summary': Dict with expected_prob, ci_5, ci_95, min, max
                - 'inputs': Dict with P1, P2, P3, P4, P5, P6 arrays/values
                - 'sensitivity': Dict with Spearman correlations for stochastic inputs
                - 'config': The SimulationConfig used
        """
        np.random.seed(self.config.seed)
        cfg = self.config

        # ─── Step 1: Outbreak probability ───
        if cfg.P1_override is not None:
            P1 = cfg.P1_override
        else:
            P1 = 1.0 - np.exp(-cfg.t * cfg.lambda_rate)

        # ─── Step 2: Infection rate in humans ───
        if cfg.P2_override is not None:
            P2 = np.full(cfg.n_samples, cfg.P2_override)
        else:
            P2 = np.random.beta(cfg.beta_alpha, cfg.beta_beta, cfg.n_samples)

        # ─── Step 3: False Negative in Bangladesh ───
        if cfg.P3_override is not None:
            P3 = np.full(cfg.n_samples, cfg.P3_override)
        else:
            P3_sensitivity = np.random.uniform(
                cfg.sensitivity_bangladesh_low, 
                cfg.sensitivity_bangladesh_high, 
                cfg.n_samples
            )
            P3 = 1.0 - P3_sensitivity  # False Negative = 1 - Sensitivity

        # ─── Step 4: False Negative in India ───
        if cfg.P4_override is not None:
            P4 = np.full(cfg.n_samples, cfg.P4_override)
        else:
            P4_sensitivity = np.random.uniform(
                cfg.sensitivity_india_low, 
                cfg.sensitivity_india_high, 
                cfg.n_samples
            )
            P4 = 1.0 - P4_sensitivity  # False Negative = 1 - Sensitivity

        # ─── Step 5: Exposure risk ───
        P5 = cfg.P5  # Already handles override via property

        # ─── Step 6: Mortality ───
        P6 = cfg.P6  # Already handles override via property

        # ─── Monte Carlo compound probability ───
        P_final = P1 * P2 * P3 * P4 * P5 * P6

        # ─── Summary statistics ───
        summary = {
            "expected_prob": float(np.mean(P_final)),
            "median": float(np.median(P_final)),
            "std_dev": float(np.std(P_final)),
            "ci_5": float(np.percentile(P_final, 5.0)),
            "ci_95": float(np.percentile(P_final, 95.0)),
            "ci_25": float(np.percentile(P_final, 25.0)),
            "ci_75": float(np.percentile(P_final, 75.0)),
            "min": float(np.min(P_final)),
            "max": float(np.max(P_final)),
        }

        # ─── Sensitivity analysis (Spearman rank correlation) ───
        sensitivity = self._sensitivity_analysis(P_final, P2, P3, P4)

        # ─── Track which values are overridden ───
        overrides_used = {
            "P1": cfg.P1_override is not None,
            "P2": cfg.P2_override is not None,
            "P3": cfg.P3_override is not None,
            "P4": cfg.P4_override is not None,
            "P5": cfg.P5_override is not None,
            "P6": cfg.P6_override is not None,
        }

        # ─── Assemble results ───
        self._results = {
            "samples": P_final,
            "summary": summary,
            "inputs": {
                "P1": P1,
                "P2": P2,
                "P3": P3,
                "P4": P4,
                "P5": P5,
                "P6": P6,
            },
            "input_means": {
                "P1": float(P1) if isinstance(P1, (int, float)) else float(np.mean(P1)),
                "P2": float(np.mean(P2)),
                "P3": float(np.mean(P3)),
                "P4": float(np.mean(P4)),
                "P5": float(P5),
                "P6": float(P6),
            },
            "sensitivity": sensitivity,
            "config": cfg,
            "lambda_rate": cfg.lambda_rate,
            "overrides_used": overrides_used,
        }

        return self._results

    @staticmethod
    def _sensitivity_analysis(P_final: np.ndarray, 
                               P2: np.ndarray, 
                               P3: np.ndarray, 
                               P4: np.ndarray) -> Dict[str, float]:
        """Perform sensitivity analysis using Spearman's rank correlation.
        
        Measures how strongly each stochastic input variable influences 
        the final risk output.
        
        Args:
            P_final: Array of final compound risk probabilities.
            P2: Array of infection rate samples (Beta distribution).
            P3: Array of false negative rate samples (Bangladesh).
            P4: Array of false negative rate samples (India).
            
        Returns:
            Dictionary mapping factor names to their Spearman ρ values.
        """
        factors = {
            "P2 (Infection Rate)": P2,
            "P3 (FN Bangladesh)": P3,
            "P4 (FN India)": P4,
        }

        correlations = {}
        for name, values in factors.items():
            rho, p_value = spearmanr(values, P_final)
            correlations[name] = {
                "rho": float(rho),
                "p_value": float(p_value),
            }

        return correlations

    @property
    def results(self) -> Optional[Dict[str, Any]]:
        """Access the last simulation results (None if not yet run)."""
        return self._results

    def __repr__(self) -> str:
        status = "run" if self._results else "not run"
        return (f"CholeraSimulator(n_samples={self.config.n_samples}, "
                f"seed={self.config.seed}, status={status})")


# ═══════════════════════════════════════════════════════════════════
# Convenience function for quick one-shot usage
# ═══════════════════════════════════════════════════════════════════

def quick_risk_assessment(n_samples: int = 10000, seed: int = 42,
                          P1: Optional[float] = None,
                          P2: Optional[float] = None,
                          P3: Optional[float] = None,
                          P4: Optional[float] = None,
                          P5: Optional[float] = None,
                          P6: Optional[float] = None) -> Dict[str, Any]:
    """Run a cholera risk assessment with optional P-value overrides.
    
    This is a convenience function for quick, one-off simulations.
    For repeated use or customized parameters, use CholeraSimulator directly.
    
    Args:
        n_samples: Number of Monte Carlo samples.
        seed: Random seed for reproducibility.
        P1: Override outbreak probability (default: Poisson-derived).
        P2: Override infection rate (default: Beta distribution sample).
        P3: Override false negative Bangladesh (default: Uniform sample).
        P4: Override false negative India (default: Uniform sample).
        P5: Override exposure risk (default: 0.11 × 0.01).
        P6: Override mortality rate (default: 0.03).
        
    Returns:
        Dictionary of simulation results (same format as CholeraSimulator.run_simulation).
    
    Example:
        >>> results = quick_risk_assessment(n_samples=50000)
        >>> print(f"Risk: {results['summary']['expected_prob']:.6e}")
        
        >>> # With custom P values
        >>> results = quick_risk_assessment(P1=0.95, P5=0.002, P6=0.05)
    """
    sim = CholeraSimulator(n_samples=n_samples, seed=seed,
                           P1=P1, P2=P2, P3=P3, P4=P4, P5=P5, P6=P6)
    return sim.run_simulation()
