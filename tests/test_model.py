"""
Unit Tests for the Cholera Risk Assessment Model

Tests ensure mathematical correctness, reproducibility, and 
consistency of the Monte Carlo simulation engine.

Run with:
    cd colera/
    python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cholera_risk.model import CholeraSimulator, SimulationConfig, quick_risk_assessment


class TestSimulationConfig:
    """Tests for the SimulationConfig dataclass."""

    def test_default_lambda_rate(self):
        """λ should equal 2779 / (22 * 12) ≈ 10.527."""
        cfg = SimulationConfig()
        expected = 2779 / (22 * 12)
        assert abs(cfg.lambda_rate - expected) < 1e-10

    def test_default_P5(self):
        """P5 = 0.11 * 0.01 = 0.0011."""
        cfg = SimulationConfig()
        assert abs(cfg.P5 - 0.0011) < 1e-10

    def test_default_P6(self):
        """P6 = 0.03 (case fatality rate)."""
        cfg = SimulationConfig()
        assert cfg.P6 == 0.03

    def test_custom_config(self):
        """Custom parameters should override defaults."""
        cfg = SimulationConfig(
            total_outbreaks=1000,
            observation_years=10,
            observation_months_per_year=12,
        )
        assert abs(cfg.lambda_rate - 1000 / 120) < 1e-10


class TestCholeraSimulator:
    """Tests for the CholeraSimulator class."""

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        sim1 = CholeraSimulator(n_samples=1000, seed=42)
        r1 = sim1.run_simulation()

        sim2 = CholeraSimulator(n_samples=1000, seed=42)
        r2 = sim2.run_simulation()

        np.testing.assert_array_equal(r1["samples"], r2["samples"])
        assert r1["summary"]["expected_prob"] == r2["summary"]["expected_prob"]

    def test_different_seeds_differ(self):
        """Different seeds should produce different results."""
        sim1 = CholeraSimulator(n_samples=1000, seed=42)
        r1 = sim1.run_simulation()

        sim2 = CholeraSimulator(n_samples=1000, seed=99)
        r2 = sim2.run_simulation()

        assert r1["summary"]["expected_prob"] != r2["summary"]["expected_prob"]

    def test_output_structure(self):
        """Results dictionary should have all expected keys."""
        sim = CholeraSimulator(n_samples=500, seed=1)
        results = sim.run_simulation()

        assert "samples" in results
        assert "summary" in results
        assert "inputs" in results
        assert "sensitivity" in results
        assert "config" in results
        assert "lambda_rate" in results
        assert "input_means" in results

    def test_sample_count(self):
        """Number of samples should match n_samples."""
        n = 5000
        sim = CholeraSimulator(n_samples=n, seed=1)
        results = sim.run_simulation()
        assert len(results["samples"]) == n

    def test_summary_statistics_keys(self):
        """Summary should contain all required statistics."""
        sim = CholeraSimulator(n_samples=500, seed=1)
        results = sim.run_simulation()
        
        required_keys = [
            "expected_prob", "median", "std_dev",
            "ci_5", "ci_95", "ci_25", "ci_75",
            "min", "max"
        ]
        for key in required_keys:
            assert key in results["summary"], f"Missing key: {key}"

    def test_probabilities_are_positive(self):
        """All simulated probabilities should be positive."""
        sim = CholeraSimulator(n_samples=5000, seed=42)
        results = sim.run_simulation()
        assert np.all(results["samples"] > 0)

    def test_probabilities_less_than_one(self):
        """All simulated probabilities should be less than 1."""
        sim = CholeraSimulator(n_samples=5000, seed=42)
        results = sim.run_simulation()
        assert np.all(results["samples"] < 1.0)

    def test_mean_within_ci(self):
        """Expected probability should be within the 5th-95th percentile range."""
        sim = CholeraSimulator(n_samples=10000, seed=42)
        results = sim.run_simulation()
        s = results["summary"]
        assert s["ci_5"] <= s["expected_prob"] <= s["ci_95"]

    def test_min_max_ordering(self):
        """Min < CI_5 < Median < CI_95 < Max."""
        sim = CholeraSimulator(n_samples=10000, seed=42)
        results = sim.run_simulation()
        s = results["summary"]
        assert s["min"] <= s["ci_5"]
        assert s["ci_5"] <= s["median"]
        assert s["median"] <= s["ci_95"]
        assert s["ci_95"] <= s["max"]

    def test_P1_near_one(self):
        """P1 (outbreak probability) should be very close to 1 given high λ."""
        sim = CholeraSimulator(n_samples=100, seed=1)
        results = sim.run_simulation()
        P1 = results["inputs"]["P1"]
        assert P1 > 0.999, f"P1 should be near 1.0, got {P1}"

    def test_P2_beta_distribution_mean(self):
        """Mean of P2 ~ Beta(1605, 24618) should be near 1605/26223 ≈ 0.0612."""
        sim = CholeraSimulator(n_samples=50000, seed=42)
        results = sim.run_simulation()
        p2_mean = results["input_means"]["P2"]
        expected_mean = 1605 / (1605 + 24618)
        assert abs(p2_mean - expected_mean) < 0.002, \
            f"P2 mean {p2_mean} not close to theoretical {expected_mean}"

    def test_P3_uniform_range(self):
        """P3 values should be in [1-0.906, 1-0.549] = [0.094, 0.451]."""
        sim = CholeraSimulator(n_samples=5000, seed=42)
        results = sim.run_simulation()
        P3 = results["inputs"]["P3"]
        assert np.all(P3 >= 0.094 - 1e-10)
        assert np.all(P3 <= 0.451 + 1e-10)

    def test_P4_uniform_range(self):
        """P4 values should be in [1-0.999, 1-0.884] = [0.001, 0.116]."""
        sim = CholeraSimulator(n_samples=5000, seed=42)
        results = sim.run_simulation()
        P4 = results["inputs"]["P4"]
        assert np.all(P4 >= 0.001 - 1e-10)
        assert np.all(P4 <= 0.116 + 1e-10)

    def test_sensitivity_analysis_keys(self):
        """Sensitivity analysis should include P2, P3, P4."""
        sim = CholeraSimulator(n_samples=1000, seed=42)
        results = sim.run_simulation()
        sens = results["sensitivity"]
        assert "P2 (Infection Rate)" in sens
        assert "P3 (FN Bangladesh)" in sens
        assert "P4 (FN India)" in sens

    def test_sensitivity_rho_range(self):
        """All Spearman ρ values should be in [-1, 1]."""
        sim = CholeraSimulator(n_samples=5000, seed=42)
        results = sim.run_simulation()
        for name, data in results["sensitivity"].items():
            assert -1.0 <= data["rho"] <= 1.0, \
                f"ρ for {name} is {data['rho']}, not in [-1, 1]"

    def test_sensitivity_p2_positive(self):
        """P2 should have a positive correlation with final risk."""
        sim = CholeraSimulator(n_samples=10000, seed=42)
        results = sim.run_simulation()
        rho = results["sensitivity"]["P2 (Infection Rate)"]["rho"]
        assert rho > 0, f"Expected positive ρ for P2, got {rho}"

    def test_repr(self):
        """String representation should be informative."""
        sim = CholeraSimulator(n_samples=1000, seed=42)
        assert "not run" in repr(sim)
        sim.run_simulation()
        assert "run" in repr(sim)


class TestQuickRiskAssessment:
    """Tests for the convenience function."""

    def test_returns_dict(self):
        """quick_risk_assessment should return a dictionary."""
        results = quick_risk_assessment(n_samples=500, seed=1)
        assert isinstance(results, dict)

    def test_matches_simulator(self):
        """Convenience function should produce same results as class."""
        results_func = quick_risk_assessment(n_samples=1000, seed=42)
        
        sim = CholeraSimulator(n_samples=1000, seed=42)
        results_class = sim.run_simulation()
        
        np.testing.assert_array_equal(
            results_func["samples"], 
            results_class["samples"]
        )


class TestCustomConfig:
    """Tests for using custom SimulationConfig."""

    def test_custom_beta_params(self):
        """Custom Beta parameters should change P2 distribution."""
        cfg = SimulationConfig(beta_alpha=100, beta_beta=900)
        sim = CholeraSimulator(n_samples=5000, seed=42, config=cfg)
        results = sim.run_simulation()
        
        p2_mean = results["input_means"]["P2"]
        expected_mean = 100 / (100 + 900)
        assert abs(p2_mean - expected_mean) < 0.01

    def test_custom_sensitivity_bounds(self):
        """Custom sensitivity bounds should change P3/P4 ranges."""
        cfg = SimulationConfig(
            sensitivity_bangladesh_low=0.8,
            sensitivity_bangladesh_high=0.9,
        )
        sim = CholeraSimulator(n_samples=5000, seed=42, config=cfg)
        results = sim.run_simulation()
        
        P3 = results["inputs"]["P3"]
        assert np.all(P3 >= 0.1 - 1e-10)
        assert np.all(P3 <= 0.2 + 1e-10)

    def test_zero_mortality(self):
        """Setting P6 = 0 should make all final probabilities zero."""
        cfg = SimulationConfig(case_fatality_rate=0.0)
        sim = CholeraSimulator(n_samples=100, seed=42, config=cfg)
        results = sim.run_simulation()
        assert np.all(results["samples"] == 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
