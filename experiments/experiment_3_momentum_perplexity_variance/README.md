# Experiment 22: Momentum Perplexity Trajectory Averages Analysis

## Overview

This experiment analyzes trajectory averages of momentum² and ln(point_perplexity) from trajectory data files. It computes the coefficient of variation across trajectories and examines the distribution of momentum² + ln(point_perplexity). This analysis helps understand the variability and relationships between these key quantities in language model dynamics at the trajectory level.

## What It Does

For each trajectory in a dataset, the analyzer:

1. **Computes trajectory averages**: Calculates the average momentum² and average ln(point_perplexity) for each trajectory
2. **Analyzes cross-trajectory distributions**: Examines how these averages vary across different trajectories
3. **Calculates coefficient of variation**: Computes CV across trajectories (not within trajectories)
4. **Examines combined quantity**: Analyzes the distribution of momentum² + ln(point_perplexity)
5. **Studies relationships**: Analyzes correlations between trajectory averages

## Key Metrics

The analyzer computes several important metrics:

- **Trajectory Averages**: For each trajectory:
  - Average momentum² across all steps
  - Average ln(point_perplexity) across all steps  
  - Sum: momentum² + ln(point_perplexity)

- **Cross-trajectory Coefficient of Variation**: 
  - CV of momentum² averages across trajectories
  - CV of ln(point_perplexity) averages across trajectories
  - CV of the sum across trajectories

- **Distribution Analysis**:
  - Statistical summaries of trajectory averages
  - Correlation between momentum² and ln(point_perplexity) trajectory averages

## Input Format

The analyzer expects JSON files with the following structure:

```json
[
  {
    "text": "Sample text...",
    "token_sequence": [1234, 5678, ...],
    "steps": [
      {
        "step": 0,
        "momentum": 0.0,
        "point_perplexity": 13.7080895540894,
        "distribution_perplexity": 297.85296630859375,
        "token": "An"
      },
      {
        "step": 1,
        "momentum": "131.81425",
        "point_perplexity": 2.17338187506797,
        "distribution_perplexity": 5.394489765167236,
        "token": "arch"
      },
      ...
    ]
  },
  ...
]
```

## Usage

### Basic Usage

```bash
# Run analysis on a specific trajectory file
python momentum_perplexity_variance_analyzer.py path/to/trajectories.json

# Specify custom output directory
python momentum_perplexity_variance_analyzer.py path/to/trajectories.json --output_dir custom_results/
```

### Example Script

```bash
# Run the example script (uses trajectories.json from experiment_1)
python run_example.py

# Run with a custom file
python run_example.py path/to/your/trajectories.json
```

### Programmatic Usage

```python
from momentum_perplexity_variance_analyzer import MomentumPerplexityVarianceAnalyzer

# Create analyzer
analyzer = MomentumPerplexityVarianceAnalyzer(
    trajectories_file="path/to/trajectories.json",
    output_dir="results/"
)

# Run full analysis
results_df = analyzer.run_full_analysis()

# Access cross-trajectory CV values
print(f"Momentum² CV: {results_df.attrs['momentum_squared_cv']:.4f}")
print(f"ln(Perplexity) CV: {results_df.attrs['ln_perplexity_cv']:.4f}")
print(f"Sum CV: {results_df.attrs['sum_cv']:.4f}")

# Access trajectory averages
print(f"Mean momentum² across trajectories: {results_df['avg_momentum_squared'].mean():.4f}")
```

## Output Files

The analysis generates several output files:

1. **`trajectory_summary.csv`**: Detailed statistics for each trajectory including:
   - Trajectory index and number of steps
   - Average momentum², average ln(point_perplexity), and their sum for each trajectory

2. **`trajectory_distributions.png`**: Four-panel plot showing:
   - Distribution of average momentum² across trajectories
   - Distribution of average ln(point_perplexity) across trajectories  
   - Distribution of momentum² + ln(point_perplexity) across trajectories
   - Coefficient of variation comparison bar chart

3. **`trajectory_samples.png`**: Line plots showing momentum² and ln(perplexity) evolution over steps for sample trajectories

4. **`correlation_analysis.png`**: Scatter plot showing correlation between trajectory averages of momentum² and ln(perplexity)

5. **`analysis_report.txt`**: Comprehensive text report with summary statistics and correlations

## Interpretation

### Coefficient of Variation (Cross-trajectory)
- **Low CV (< 0.5)**: Similar trajectory averages across the dataset - consistent behavior
- **Medium CV (0.5-1.0)**: Moderate variability in trajectory averages
- **High CV (> 1.0)**: High variability in trajectory averages - diverse trajectory behaviors

### Distribution Analysis
- **Momentum² + ln(point_perplexity)**: Combined quantity that may represent total "information content"
- **Correlation**: Relationship between momentum and perplexity at the trajectory level

### Trajectory-level Insights
- **Consistent trajectories**: Low CV values suggest similar average behavior across trajectories
- **Diverse trajectories**: High CV values suggest different trajectory "types" or regimes
- **Correlation patterns**: Positive/negative correlations reveal systematic relationships

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation and analysis
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical visualization (optional, for enhanced plots)

## Example Output

```
Momentum Perplexity Trajectory Averages Analysis
================================================
Loaded 150 trajectories from trajectories.json

TRAJECTORY AVERAGES ANALYSIS SUMMARY
====================================
Number of trajectories analyzed: 150
Average trajectory length: 45.3 steps

Coefficient of Variation (across trajectories):
  Momentum²: 0.6234
  ln(Point Perplexity): 0.2456
  Sum (momentum² + ln(perplexity)): 0.5123

Correlation between trajectory averages: 0.3456
  → Moderate correlation detected.
```

## Related Experiments

This experiment complements:
- **Experiment 19**: PPL Momentum Fit - Provides the trajectory data
- **Experiment 5**: Energy Conservation - Studies conserved quantities in similar systems
- **Experiment 14**: Semantic Coherence - Analyzes semantic properties of trajectories

## Key Differences from Within-trajectory Analysis

This experiment focuses on **cross-trajectory** patterns rather than **within-trajectory** patterns:
- Computes averages per trajectory, then analyzes how these averages vary across trajectories
- CV measures variability between different trajectories, not variability within individual trajectories
- Useful for understanding whether different texts/contexts lead to systematically different momentum/perplexity patterns 