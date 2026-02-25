# 🦠 Cholera Risk Assessment — Monte Carlo Simulation

A Python package for **quantitative risk analysis** of transboundary cholera infection risk using stochastic **Monte Carlo simulation**. All parameters (P1–P6) are fully configurable to support any region or scenario.

---

## 📐 Model Overview

The model computes the compound probability of cholera-related mortality through a chain of six risk factors:

```
P_final = P1 × P2 × P3 × P4 × P5 × P6
```

| Step | Parameter | Description | Default Distribution |
|------|-----------|-------------|---------------------|
| P1 | Outbreak Probability | Probability of cholera outbreak in source region | Poisson: `1 − e^(−λt)`, λ = 2779/(22×12) |
| P2 | Infection Rate | Human cholera infection probability | Beta(1605, 24618) |
| P3 | False Negative (Source) | Missed cases in source region | 1 − Uniform(0.549, 0.906) |
| P4 | False Negative (Target) | Missed cases in target region | 1 − Uniform(0.884, 0.999) |
| P5 | Exposure Risk | Unsafe water × unsafe sanitation | 0.11 × 0.01 = 0.0011 |
| P6 | Mortality | Case fatality rate | 0.03 |

> **Note:** All default values above can be overridden. Pass custom P1–P6 values to adapt the model to any region or scenario.

The simulation generates **10,000** (configurable) random samples and computes:
- Expected Probability (mean)
- 90% Confidence Interval (5th–95th percentile)
- Min/Max range
- Sensitivity analysis via **Spearman's rank correlation**

---

## 📦 Installation

```bash
# Install from PyPI
pip install cholera-risk

# Or install a specific version
pip install cholera-risk==0.2.0
```

---

## 🚀 Quick Start

### Using the Class API

```python
from cholera_risk import CholeraSimulator

# Create simulator with 10,000 samples
sim = CholeraSimulator(n_samples=10000, seed=42)

# Run the Monte Carlo simulation
results = sim.run_simulation()

# Access results
print(f"Expected Risk: {results['summary']['expected_prob']:.6e}")
print(f"90% CI: [{results['summary']['ci_5']:.2e}, {results['summary']['ci_95']:.2e}]")
```

### Using the Convenience Function

```python
from cholera_risk.model import quick_risk_assessment

results = quick_risk_assessment(n_samples=50000, seed=123)
print(f"Risk: {results['summary']['expected_prob']:.6e}")
```

### Generating Visualizations

```python
from cholera_risk import CholeraSimulator
from cholera_risk.visualization import plot_all, print_summary

sim = CholeraSimulator(n_samples=10000, seed=42)
results = sim.run_simulation()

# Print formatted summary
print_summary(results)

# Generate all plots (histogram, sensitivity, scatter)
plot_all(results, save_dir="./output_plots/")
```

### Custom P Values (v0.2.0+)

```python
from cholera_risk import CholeraSimulator

# Override specific P values
sim = CholeraSimulator(
    n_samples=10000, seed=42,
    P1=0.95,       # Custom outbreak probability
    P5=0.002,      # Custom exposure risk
    P6=0.05,       # Custom mortality rate
)
results = sim.run_simulation()

# Override ALL P values
sim = CholeraSimulator(
    P1=0.99, P2=0.06, P3=0.27, P4=0.06, P5=0.0011, P6=0.03
)
results = sim.run_simulation()
```

### Advanced: Custom Distribution Parameters

```python
from cholera_risk.model import CholeraSimulator, SimulationConfig

# Customize underlying distributions for a different region
config = SimulationConfig(
    total_outbreaks=3000,
    observation_years=25,
    beta_alpha=2000,
    beta_beta=30000,
    sensitivity_bangladesh_low=0.60,   # Source region sensitivity
    sensitivity_bangladesh_high=0.95,
    sensitivity_india_low=0.80,        # Target region sensitivity
    sensitivity_india_high=0.99,
    unsafe_water_fraction=0.15,
    case_fatality_rate=0.05,
)

sim = CholeraSimulator(n_samples=20000, seed=99, config=config)
results = sim.run_simulation()
```

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=cholera_risk --cov-report=term-missing
```

---

## 📁 Directory Structure

```
cholera/
├── cholera_risk/              # The Python package
│   ├── __init__.py            # Package init, exports CholeraSimulator
│   ├── model.py               # Core simulation logic (math & stats)
│   └── visualization.py       # Plotting functions (matplotlib/seaborn)
├── tests/                     # Unit tests (26 tests)
│   ├── __init__.py
│   └── test_model.py
├── README.md                  # This file
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
└── setup.py                   # Package installation script
```

---

## 📊 Output Example

```
=================================================================
   CHOLERA RISK ASSESSMENT — Monte Carlo Simulation Results
=================================================================
  Samples:              10,000
  Random Seed:          42
  λ (Poisson rate):     10.5265
-----------------------------------------------------------------
  Expected Probability: 3.207426e-08
  Median:               2.450000e-08
  90% CI:               [3.16e-09, 7.74e-08]
  Minimum:              2.67e-10
  Maximum:              1.05e-07
-----------------------------------------------------------------
  Sensitivity Analysis (Spearman ρ):
    P2 (Infection Rate):  ρ = 0.0397
    P3 (FN Source):       ρ = 0.4899
    P4 (FN Target):       ρ = 0.8312
=================================================================
```

---

## 🔬 Methodology

- **Monte Carlo Simulation**: Generates thousands of random scenarios by sampling each stochastic input from its probability distribution
- **Beta Distribution (P2)**: Models the uncertainty in human infection rates using Bayesian parameters derived from epidemiological data
- **Uniform Distributions (P3, P4)**: Represent the range of CholKit RDT diagnostic sensitivity
- **Spearman Rank Correlation**: Non-parametric measure of how strongly each input drives variations in the output

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👥 Authors

- **Navnath Kamble** — [navnathkamble0007@gmail.com](mailto:navnathkamble0007@gmail.com)
- **Yamini Madugu** — [maduguyamini63662@gmail.com](mailto:maduguyamini63662@gmail.com)
