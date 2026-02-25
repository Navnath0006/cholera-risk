import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr

def cholera_risk_assessment(n_samples=10000, seed=42):
    # Set seed for reproducibility
    np.random.seed(seed)
   
    # Given Parameters
    t = 1
    lambda_rate = 2779 / (22 * 12)  # From table
   
    # Step 1: Probability of Cholera outbreak in Bangladesh
    P1 = 1 - np.exp(-t * lambda_rate)
   
    # Step 2: Probability of Cholera infection in humans in Bangladesh (Beta distribution)
    P2 = np.random.beta(1605, 24618, n_samples)
    #P2 = np.random.beta(109052, 66386158, n_samples)
   
    # Step 3: False Negative clinical examination of infected individuals (1 - Sensitivity of Cholkit RDT)
    P3a = np.random.uniform(0.549, 0.906, n_samples)
    P3 = 1 - P3a
   
    # Step 4: FN clinical examination in India (1 - Sensitivity of Cholkit RDT)
    P4a = np.random.uniform(0.884, 0.999, n_samples)
    P4 = 1 - P4a
   
    # Step 5: Exposure risk due to unsafe water and sanitation access
    P5a = 0.11
    P5b = 0.01
    P5 = P5a * P5b
   
    # Step 6: Probability of mortality
    P6 = 0.03
   
    # Monte Carlo Simulation
    P_final_samples = P1 * P2 * P3 * P4 * P5 * P6
   
    # Statistics
    expected_prob = np.mean(P_final_samples)
    confidence_interval = np.percentile(P_final_samples, [5.0, 95.0])
    min_val = np.min(P_final_samples)
    max_val = np.max(P_final_samples)
   
    return P_final_samples, expected_prob, confidence_interval, min_val, max_val, P1, P2, P3, P4, P5, P6

def plot_distributions(P_final_samples, expected_prob, min_val, max_val):
    """Plots the probability distribution of the final risk assessment."""
    plt.figure(figsize=(10, 6))
    sns.histplot(P_final_samples, bins=50, kde=True, color='skyblue')
   
    # Add lines for mean, min, and max
    plt.axvline(expected_prob, color='red', linestyle='--', label=f'Mean = {expected_prob:.2e}')
    plt.axvline(min_val, color='green', linestyle='--', label=f'Min = {min_val:.2e}')
    plt.axvline(max_val, color='purple', linestyle='--', label=f'Max = {max_val:.2e}')
   
    plt.xlabel("Probability of Cholera Infection in India from Bangladesh")
    plt.ylabel("Frequency")
    plt.title("Distribution of Simulated Risk Probabilities")
    plt.legend()
    plt.tight_layout()
    plt.show()

def sensitivity_analysis(P_final_samples, P2, P3, P4):
    """Performs sensitivity analysis using Spearman's rank correlation."""
    factors = {
        'P2': P2,
        'P3': P3,
        'P4': P4
    }
   
    correlations = {}
    for key, values in factors.items():
        correlation, _ = spearmanr(values, P_final_samples)
        correlations[key] = correlation
   
    # Convert to DataFrame for plotting
    df_correlation = pd.DataFrame.from_dict(correlations, orient='index', columns=['Spearman Correlation'])
    df_correlation = df_correlation.sort_values(by='Spearman Correlation', ascending=False)
    print(df_correlation)
   
    # Bar Plot with correlation values on bars
    plt.figure(figsize=(9.5, 5))
    bars = plt.barh(df_correlation.index, df_correlation['Spearman Correlation'], color='tomato')
    plt.xlabel("Spearman Rank Correlation coefficient")
    plt.title("Sensitivity Analysis")
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01*np.sign(width), bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}', va='center', ha='left' if width > 0 else 'right')
    plt.tight_layout()
    plt.show()

# Running the model
P_final_samples, expected_prob, conf_interval, min_val, max_val, P1, P2, P3, P4, P5, P6 = cholera_risk_assessment()
print(f"Expected Probability: {expected_prob:.6e}")
print(f"95% Confidence Interval: [{conf_interval[0]:.2e}, {conf_interval[1]:.2e}]")
print(f"Minimum: {min_val:.2e}, Maximum: {max_val:.2e}")

# Visualization
plot_distributions(P_final_samples, expected_prob, min_val, max_val)
sensitivity_analysis(P_final_samples, P2, P3, P4)