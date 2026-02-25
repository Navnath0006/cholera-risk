"""
Cholera Risk Assessment - Monte Carlo Simulation Package

A quantitative risk analysis tool for evaluating transboundary cholera 
infection risk from Bangladesh to India using stochastic Monte Carlo simulation.

Usage:
    from cholera_risk import CholeraSimulator

    sim = CholeraSimulator(n_samples=10000, seed=42)
    results = sim.run_simulation()
    print(results["summary"])
"""

from cholera_risk.model import CholeraSimulator

__version__ = "1.0.0"
__author__ = "Navnath Kamble, Yamini Madugu"
__all__ = ["CholeraSimulator"]
