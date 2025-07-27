# Experiment 11.0: Momentum-Perplexity Band Mapping

## Overview

This experiment systematically maps the **joint distribution of momentum and perplexity** in language model hidden space, treating text generation as a dynamical system. It explores the relationship between momentum (rate of change in hidden space) and perplexity (uncertainty/confidence) to understand the fundamental dynamics of language model trajectories.

## Python Files

### Core Experiment Scripts

**1. `momentum_perplexity_mapping_experiment.py` (Main Mapping - 1106 lines)**
- **Purpose**: Systematic mapping of momentum-perplexity joint distributions
- **Function**: Samples across predefined momentum ranges and angles to create comprehensive mapping
- **Features**:
  - Joint distribution analysis of momentum vs perplexity
  - Systematic sampling across momentum ranges (50-100, 100-150, 150-200, 200-300)
  - Multiple angle bands (5°, 10°, 15°, 20°, 30°, 45°)
  - Both natural trajectory extraction and artificial sampling
  - Advanced 3D visualizations showing density distributions
  - Smart perplexity cutoff handling for cleaner visualizations
  - Class: `MomentumPerplexityMappingExperiment`

**2. `momentum_perplexity_phase_space_experiment.py` (Phase Space Analysis - 632 lines)**
- **Purpose**: Treats language generation as dynamical system in momentum-perplexity phase space
- **Function**: Analyzes trajectories to find attractors, cycles, and flow patterns
- **Features**:
  - 2D phase space coordinates: X-axis = momentum, Y-axis = perplexity
  - Dynamical system analysis (attractors, cycles, fixed points)
  - Token type analysis and energy measures
  - Velocity and acceleration calculations
  - Comprehensive phase space statistics
  - Uses `PhaseSpaceVisualizer` for advanced plotting
  - Class: `MomentumPerplexityPhaseSpaceExperiment`

### Analysis & Validation Scripts

**3. `test_r_squared_analysis.py` (Conservation Law Testing - 134 lines)**
- **Purpose**: Demonstrates R² analysis for conservation law discovery
- **Function**: Compares point vs distribution perplexity for conservation validation
- **Features**:
  - Tests conservation law quality using R² metrics
  - Compares point perplexity vs distribution perplexity
  - Conservation coefficient of variation analysis
  - Linear fit R² assessment
  - Uses `ConservedQuantitiesAnalyzer` for detailed analysis

## Theoretical Background

### Dynamical System Framework

This experiment treats language model generation as a **dynamical system** where:

```
Momentum = ||h_t - h_{t-1}||  (velocity in hidden space)
Perplexity = uncertainty/confidence (potential energy-like)
```

### Phase Space Coordinates
- **X-axis**: Momentum (rate of change in hidden space)
- **Y-axis**: Perplexity (point or distribution)

### Key Relationships Explored
1. **Joint Distribution**: P(momentum, perplexity) across natural and artificial samples
2. **Phase Space Dynamics**: Attractors, cycles, flow patterns
3. **Conservation Laws**: Energy-like quantities that remain constant
4. **Momentum Ranges**: Systematic sampling across 50-300 units

### Perplexity Types
- **Point Perplexity**: `1/max_probability` (focused on most likely token)
- **Distribution Perplexity**: `exp(entropy)` (full distribution uncertainty)

## Experimental Design

### Momentum-Perplexity Mapping Process
1. **Load Wikipedia samples** using datasets library
2. **Extract natural trajectories** from model hidden states
3. **Systematic artificial sampling**:
   - Multiple momentum ranges: [(50,100), (100,150), (150,200), (200,300)]
   - Multiple angle bands: [5°, 10°, 15°, 20°, 30°, 45°]
   - Configurable samples per range/angle combination
4. **Calculate both perplexity types** for each sample
5. **Create joint distributions** and marginal distributions
6. **Generate 3D visualizations** of momentum-perplexity relationships

### Phase Space Analysis Process
1. **Load Wikipedia articles** for diverse text samples
2. **Extract trajectory phase space** for each text sample
3. **Analyze dynamical properties**:
   - Attractors (regions trajectories converge to)
   - Cycles (repeated patterns)
   - Flow patterns (how trajectories move)
   - Fixed points (equilibrium states)
4. **Calculate energy measures** and velocity/acceleration
5. **Statistical analysis** of phase space coverage

### Conservation Law Testing
1. **Load existing trajectory data** from phase space analysis
2. **Test conservation laws** using R² metrics
3. **Compare perplexity types** for conservation quality
4. **Validate theoretical predictions** about conserved quantities

## Usage

### Momentum-Perplexity Mapping
```bash
# Basic mapping experiment
python momentum_perplexity_mapping_experiment.py --num_articles 20

# Custom momentum ranges and angles
python momentum_perplexity_mapping_experiment.py --momentum_ranges "50,100" "100,150" "200,300" --angle_bands 5 10 20 --samples_per_range 30

# Apply perplexity cutoff for cleaner visualization
python momentum_perplexity_mapping_experiment.py --num_articles 20 --perplexity_cutoff 100

# Quick test run
python momentum_perplexity_mapping_experiment.py --num_articles 5 --samples_per_range 10 --perplexity_cutoff 50
```

### Phase Space Analysis
```bash
# Basic phase space analysis
python momentum_perplexity_phase_space_experiment.py --num_articles 20

# Focus on specific perplexity range
python momentum_perplexity_phase_space_experiment.py --perplexity_cutoff 50 --num_articles 30

# Detailed analysis with longer trajectories
python momentum_perplexity_phase_space_experiment.py --max_steps_per_text 15 --num_articles 25
```

### Conservation Law Testing
```bash
# Test R² analysis (requires existing trajectory data)
python test_r_squared_analysis.py
```

### Parameter Options

**Mapping Experiment**:
- `--model_name`: HuggingFace model (default: "Qwen/Qwen2.5-1.5B-Instruct")
- `--device`: Device to use (default: "cpu")
- `--num_articles`: Number of Wikipedia articles (default: 20)
- `--momentum_ranges`: Momentum ranges as "min,max" strings
- `--angle_bands`: Angle bands in degrees (default: [5, 10, 20, 30])
- `--samples_per_range`: Samples per range per angle (default: 30)
- `--perplexity_cutoff`: Maximum perplexity for visualization
- `--noise_scale_method`: Noise scaling method ("tan", "linear", "sin")

**Phase Space Experiment**:
- `--model_name`: HuggingFace model
- `--device`: Device to use
- `--num_articles`: Number of articles to analyze (default: 20)
- `--max_steps_per_text`: Maximum trajectory steps per text (default: 10)
- `--perplexity_cutoff`: Maximum perplexity for visualization

## Results Structure

### Momentum-Perplexity Mapping Output
```
momentum_perplexity_mapping_YYYYMMDD_HHMMSS/
├── results_summary.json                           # Experiment summary
├── natural_momentum_values.npy                    # Natural trajectory momentum
├── natural_point_perplexities.npy                 # Natural point perplexity
├── natural_distribution_perplexities.npy          # Natural distribution perplexity
├── artificial_momentum_values.npy                 # Artificial sample momentum
├── artificial_point_perplexities.npy              # Artificial point perplexity
├── artificial_distribution_perplexities.npy       # Artificial distribution perplexity
├── momentum_perplexity_joint_distribution.png     # 3D joint distribution plots
├── natural_vs_artificial_comparison.png           # Natural vs artificial comparison
└── momentum_averaged_analysis.png                 # Momentum-averaged perplexity analysis
```

### Phase Space Analysis Output
```
phase_space_analysis_YYYYMMDD_HHMMSS/
├── phase_space_analysis.json                      # Statistical analysis results
├── trajectories.json                              # Individual trajectory data
├── momentum_data.npy                              # Raw momentum values
├── point_perplexity_data.npy                      # Raw point perplexity values
├── distribution_perplexity_data.npy               # Raw distribution perplexity values
└── phase_space_visualizations/                    # Comprehensive visualizations
    ├── trajectory_plots.png                       # Individual trajectory plots
    ├── density_heatmaps.png                       # Phase space density maps
    ├── flow_field_analysis.png                    # Flow pattern analysis
    └── attractor_analysis.png                     # Attractor identification
```

### Conservation Law Testing Output
```
r_squared_test_point/                               # Point perplexity analysis
r_squared_test_distribution/                       # Distribution perplexity analysis
r_squared_analysis.png                             # Comparison visualization
```

## Visualization

### 3D Joint Distribution Plots
1. **Momentum vs Point Perplexity**: 3D surface showing density distribution
2. **Momentum vs Distribution Perplexity**: 3D surface with alternative perplexity metric
3. **Natural vs Artificial Comparison**: Side-by-side scatter plots
4. **Momentum-Averaged Analysis**: Shows perplexity trends across momentum bins

### Phase Space Visualizations
1. **Individual Trajectories**: Each trajectory plotted in momentum-perplexity space
2. **Density Heatmaps**: Show regions of high trajectory density
3. **Flow Field Analysis**: Vector fields showing trajectory directions
4. **Attractor Analysis**: Identification of convergence points and cycles

### Conservation Law Plots
1. **R² Comparison**: Point vs distribution perplexity conservation quality
2. **Parameter Optimization**: Shows optimal conservation law parameters
3. **Coefficient of Variation**: Measures conservation law consistency

## Dependencies

### Required Packages
- `torch` and `transformers` for language model access
- `numpy` and `scipy` for numerical computations
- `matplotlib` and `seaborn` for basic plotting
- `datasets` for Wikipedia data loading
- `json` and `pathlib` for data handling

### External Components
- `phase_space_visualizer.py`: Advanced phase space visualization library
- `conserved_quantities_analyzer.py`: Conservation law analysis toolkit

## Expected Results

### Momentum-Perplexity Relationships
- **Natural trajectories**: Show characteristic momentum-perplexity patterns
- **Artificial sampling**: Maps full joint distribution space
- **Range effects**: Different behavior across momentum ranges (50-300)
- **Angle sensitivity**: Directional effects on perplexity relationships

### Phase Space Dynamics
- **Attractors**: Regions where trajectories tend to converge
- **Flow patterns**: Systematic movement through momentum-perplexity space
- **Energy conservation**: Conserved quantities in the dynamical system
- **Token type effects**: Different dynamics for function words vs content words

### Conservation Laws
- **Point perplexity**: Expected to show better conservation properties
- **R² metrics**: Should exceed 0.8 for strong conservation laws
- **Parameter stability**: Optimal α, β parameters should be consistent

## Scientific Significance

### Understanding Language Model Dynamics
1. **Momentum-perplexity coupling**: How hidden space movement relates to uncertainty
2. **Phase space structure**: Reveals underlying dynamics of text generation
3. **Conservation laws**: Identifies fundamental invariants in the system
4. **Predictive relationships**: How momentum can predict perplexity and vice versa

### Applications
- **Generation control**: Using momentum-perplexity relationships for steering
- **Quality assessment**: Perplexity as indicator of trajectory health
- **Model analysis**: Understanding internal dynamics across different models
- **Anomaly detection**: Identifying unusual momentum-perplexity combinations

### Theoretical Implications
- **Statistical mechanics**: Validates thermodynamic approach to language models
- **Dynamical systems**: Demonstrates phase space analysis applicability
- **Energy conservation**: Tests fundamental conservation principles
- **Momentum homeostasis**: Evidence for optimal momentum ranges

## Connection to Other Experiments

This experiment builds on and extends:
- **Experiments 1-3**: Maxwell-Boltzmann distributions, phase transitions, temperature scaling
- **Experiment 9**: Momentum ranking theory validation
- **Other steering experiments**: Momentum-based control methods

And provides foundation for:
- **Advanced steering techniques**: Using momentum-perplexity relationships
- **Conservation-based methods**: Leveraging discovered conservation laws
- **Predictive models**: Momentum-to-perplexity and perplexity-to-momentum prediction

## Troubleshooting

### Common Issues
- **Memory constraints**: Large momentum ranges and sample counts require significant memory
- **Visualization performance**: 3D plots with many points can be slow
- **Perplexity outliers**: Use `--perplexity_cutoff` to handle extreme values
- **Missing dependencies**: Ensure `phase_space_visualizer` and `conserved_quantities_analyzer` are available

### Performance Tips
- Start with smaller parameter sets for testing
- Use perplexity cutoffs to improve visualization quality
- Consider GPU acceleration for larger experiments
- Monitor memory usage with large sample counts

### Data Quality
- Check for reasonable momentum and perplexity ranges
- Validate that both natural and artificial sampling produce data
- Ensure trajectory extraction succeeds for most text samples
- Verify conservation law R² values are meaningful (> 0.4) 