# Physics-Inspired Language Model Analysis Experiments

This repository contains a collection of experiments analyzing language models through the lens of physics concepts, treating text generation as a dynamical system. Each experiment explores different aspects of momentum, perplexity, and conservation laws in language model hidden space.

## 🚀 Quick Start

### Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

### Environment Setup

For experiments requiring LLM evaluation (Experiment 2), set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

For acessing some of the models (llama 1b and llama 3b) you will need to have a Huggingface token

### Pre-computed Trajectory Data

Pre-computed trajectory data for experiments is available on Google Drive:
- **[Experiment Trajectories](https://drive.google.com/drive/folders/1sQwmiuUPMxEKmEKivnVHZVAc9Ab3REso?usp=sharing)** - Contains trajectory data from various experiments. **Note**: Only the `phase_space_analysis*` folders are used in the present experiments, particularly for Experiment 1 analysis

## 📊 Experiments Overview

| Experiment | Description | Main Focus | Key Output |
|-----------|-------------|------------|------------|
| [Experiment 1](#experiment-1-momentum-perplexity-band-trajectory-extractor) | Momentum-Perplexity Band Mapping | Joint distribution analysis, phase space dynamics | 3D visualizations, phase space plots |
| [Experiment 2](#experiment-2-jacobian-steering-completion-evaluation) | Jacobian Steering Completion Evaluation | Steering effectiveness via LLM evaluation | Completion quality scores |
| [Experiment 3](#experiment-3-momentum-perplexity-variance) | Momentum Perplexity Trajectory Averages | Cross-trajectory coefficient of variation | Statistical distributions |
| [Experiment 4](#experiment-4-token-transition-interpolation) | Token Transition Interpolation | Decision boundary smoothness analysis | Interpolation uniqueness metrics |

---

## Experiment 1: Momentum-Perplexity Band Trajectory Extractor

**Location**: `experiments/experiment_1_momentum_perplexity_band_trajectory_extractor/`

### Overview
Systematically maps the joint distribution of momentum and perplexity in language model hidden space, treating text generation as a dynamical system. Explores phase space dynamics, attractors, and conservation laws.

### Key Features
- Joint distribution analysis of momentum vs perplexity
- Phase space analysis with attractor identification
- Conservation law testing and validation
- 3D visualizations and flow field analysis

### Pre-computed Data
Pre-computed trajectory data for this experiment is available in the [shared Google Drive folder](https://drive.google.com/drive/folders/1sQwmiuUPMxEKmEKivnVHZVAc9Ab3REso?usp=drive_link). **Use the `phase_space_analysis*` folders** from the drive, which can be used for analysis without running the full data generation pipeline.

### How to Run

#### Basic Momentum-Perplexity Mapping
```bash
cd experiments/experiment_1_momentum_perplexity_band_trajectory_extractor/

# Basic mapping experiment
python momentum_perplexity_phase_space_experiment.py --num_articles 20

# Custom momentum ranges and angles
python momentum_perplexity_phase_space_experiment.py \
  --momentum_ranges "50,100" "100,150" "200,300" \
  --angle_bands 5 10 20 \
  --samples_per_range 30

# Quick test run
python momentum_perplexity_phase_space_experiment.py \
  --num_articles 5 \
  --samples_per_range 10 \
  --perplexity_cutoff 50
```

#### Phase Space Analysis
```bash
# Basic phase space analysis
python momentum_perplexity_phase_space_experiment.py --num_articles 20

# Focus on specific perplexity range
python momentum_perplexity_phase_space_experiment.py \
  --perplexity_cutoff 50 \
  --num_articles 30

# Detailed analysis with longer trajectories
python momentum_perplexity_phase_space_experiment.py \
  --max_steps_per_text 15 \
  --num_articles 25
```

#### Run All Models (Batch Processing)
```bash
# Run across multiple models listed in test_models.txt (requires significant compute time)
cd experiments/experiment_1_momentum_perplexity_band_trajectory_extractor/
./run_all_models_phase_space.sh
```

### Key Parameters
- `--model_name`: HuggingFace model (default: "Qwen/Qwen2.5-1.5B-Instruct")
- `--num_articles`: Number of Wikipedia articles (default: 20)
- `--momentum_ranges`: Momentum ranges as "min,max" strings
- `--angle_bands`: Angle bands in degrees (default: [5, 10, 20, 30])
- `--perplexity_cutoff`: Maximum perplexity for visualization
- `--max_steps_per_text`: Maximum trajectory steps per text (default: 10)

### Available Files & Output Structure
```
experiment_1_momentum_perplexity_band_trajectory_extractor/
├── momentum_perplexity_phase_space_experiment.py  # Main experiment script (42KB)
├── phase_space_visualizer.py                      # Visualization component (57KB)
├── run_all_models_phase_space.sh                  # Batch processing script (11KB)
├── test_models.txt                                 # List of models to test (679B)
├── phase_space_analysis_20250727_133458/          # Example results directory
└── README.md                                       # Local documentation (13KB)

# After running experiments, results are saved as:
phase_space_analysis_YYYYMMDD_HHMMSS/
├── phase_space_analysis.json              # Statistical analysis results
├── trajectories.json                      # Individual trajectory data
├── momentum_data.npy                      # Raw momentum values
├── point_perplexity_data.npy             # Raw point perplexity values
├── distribution_perplexity_data.npy      # Raw distribution perplexity values
└── phase_space_visualizations/           # Comprehensive visualizations
    ├── trajectory_plots.png              # Individual trajectory plots
    ├── density_heatmaps.png              # Phase space density maps
    ├── flow_field_analysis.png           # Flow pattern analysis
    └── attractor_analysis.png            # Attractor identification
```

---

## Experiment 2: Jacobian Steering Completion Evaluation

**Location**: `experiments/experiment_2_jacobian_completion_eval/`

### Overview
Evaluates the effectiveness of Jacobian steering by comparing how well natural vs steered token distributions complete sentences from Wikipedia text using LLM-based evaluation.

### Key Features
- LLM-based completion quality evaluation
- Jacobian steering effectiveness measurement
- Statistical significance testing
- Structured evaluation responses with Pydantic models

### How to Run

#### Basic Usage
```bash
cd experiments/experiment_2_jacobian_completion_eval/

# Basic evaluation
python run_jacobian_completion_experiment.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --num_samples 50 \
  --prompt_length 20 \
  --continuation_length 1 \
  --top_n 10 \
  --device cuda \
  --output_dir ./results
```

#### Example Configurations

**Quick Test**:
```bash
cd experiments/experiment_2_jacobian_completion_eval/
python run_jacobian_completion_experiment.py --num_samples 10 --top_n 5
```

**Multi-token Evaluation**:
```bash
cd experiments/experiment_2_jacobian_completion_eval/
python run_jacobian_completion_experiment.py \
  --num_samples 50 \
  --prompt_length 15 \
  --continuation_length 3 \
  --top_n 10
```

**Large Model Evaluation**:
```bash
cd experiments/experiment_2_jacobian_completion_eval/
python run_jacobian_completion_experiment.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --num_samples 200 \
  --continuation_length 2 \
  --device cuda
```



### Key Parameters
- `--model_name`: HuggingFace model identifier
- `--num_samples`: Number of Wikipedia articles to process (default: 50)
- `--prompt_length`: Length of prompt in tokens (default: 20)
- `--continuation_length`: Length of continuation to evaluate (default: 1)
- `--top_n`: Number of top tokens to evaluate (default: 10)
- `--device`: Computing device (`cuda`, `cpu`, `mps`)

### Available Files & Output Structure
```
experiment_2_jacobian_completion_eval/
├── run_jacobian_completion_experiment.py  # Main experiment script (62KB)
├── results/                               # Output directory for results
└── README.md                              # Local documentation (9.3KB)

# After running experiments, results are saved as:
jacobian_completion_YYYYMMDD_HHMMSS/
├── completion_evaluation_results.csv      # Main results with scores
├── detailed_completion_analysis.json      # Detailed token analysis
├── completion_evaluation_analysis.png     # Comprehensive plots
├── completion_evaluation_summary.png      # Summary statistics
└── experiment_config.json                 # Configuration metadata
```

### Requirements
- **OpenAI API Key**: Required for LLM evaluation
- **GPU**: Recommended for models >1B parameters
- **Memory**: 8GB+ RAM, 6GB+ VRAM for 1.5B models

---

## Experiment 3: Momentum Perplexity Variance

**Location**: `experiments/experiment_3_momentum_perplexity_variance/`

### Overview
Analyzes trajectory averages of momentum² and ln(point_perplexity) from trajectory data files. Computes coefficient of variation across trajectories and examines distribution relationships.

### Key Features
- Cross-trajectory coefficient of variation analysis
- Trajectory-level averaging and distribution analysis
- Statistical correlation analysis
- Momentum² + ln(perplexity) combined quantity analysis

### How to Run

#### Basic Analysis
```bash
cd experiments/experiment_3_momentum_perplexity_variance/

# Existing trajectory data is available in trajectories.json (20MB)
# Various beta fitting results are available in beta_fitting_phase_space_analysis_* directories
```



#### Beta Parameter Fitting
```bash
cd experiments/experiment_3_momentum_perplexity_variance/
# Fit beta parameters for conservation laws
python fit_beta_parameter.py

# Batch beta fitting across multiple configurations
./run_beta_fitting_batch.sh
```

### Key Parameters
- `fit_beta_parameter.py`: Beta parameter fitting script
- `run_beta_fitting_batch.sh`: Batch processing script for multiple configurations
- `trajectories.json`: Pre-existing trajectory data (20MB)
- Multiple `beta_fitting_phase_space_analysis_*` result directories available

### Available Data & Output Structure
```
experiment_3_momentum_perplexity_variance/
├── trajectories.json                      # Pre-existing trajectory data (20MB)
├── measurements.json                      # Energy and derived metrics, CV values (8KB, 222 lines)
├── trajectory_paths.txt                   # Trajectory file paths (4.1KB)
├── fit_beta_parameter.py                  # Beta parameter fitting script (20KB)
├── run_beta_fitting_batch.sh              # Batch processing script (6KB)
├── README_batch_fitting.md                # Additional batch fitting documentation (3.9KB)
├── conservation_write_only_10_models_with_random_baseline.zip  # Compressed results (631MB)
├── beta_fitting_20250727_135540/          # Recent beta fitting results
├── beta_fitting_phase_space_analysis_*    # Multiple timestamped result directories
└── conservation_write_only_10_models_with_random_baseline/     # Uncompressed results
```

### Key Data Files

**`measurements.json`** (8KB, 222 lines): Contains computed energy values, derived metrics, coefficient of variation (CV), and other statistical measures from trajectory analysis. This file provides pre-computed results for:
- Energy conservation quantities
- Cross-trajectory coefficient of variation
- Statistical distributions of momentum² and ln(perplexity)
- Correlation metrics and derived analytical values

### Input Format
Expects JSON files with trajectory data structure:
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
      ...
    ]
  },
  ...
]
```

---

## Experiment 4: Token Transition Interpolation

**Location**: `experiments/experiment_4_token_transition_interpolation/`

### Overview
Analyzes how token predictions change along interpolated paths between consecutive hidden states during text generation. Examines the "smoothness" of language model decision space.

### Key Features
- Linear interpolation between consecutive hidden states
- Token prediction uniqueness counting
- Decision boundary smoothness analysis
- Model architecture comparison capabilities

### How to Run

#### Basic Usage
```bash
cd experiments/experiment_4_token_transition_interpolation/

# Basic interpolation experiment
python token_transition_interpolation_experiment.py \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --continuation_size 10 \
  --prompt_size 50 \
  --num_samples 100 \
  --k_points 10
```

#### Example Configurations

**Quick Test (Small Model)**:
```bash
cd experiments/experiment_4_token_transition_interpolation/
python token_transition_interpolation_experiment.py \
  --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
  --continuation_size 5 \
  --prompt_size 30 \
  --num_samples 10 \
  --k_points 5
```

**Full Analysis (Large Model)**:
```bash
cd experiments/experiment_4_token_transition_interpolation/
python token_transition_interpolation_experiment.py \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --continuation_size 15 \
  --prompt_size 100 \
  --num_samples 200 \
  --k_points 20 \
  --top_k_tokens 3
```



### Key Parameters
- `--model`: HuggingFace model identifier
- `--continuation_size`: Number of tokens to generate per sample
- `--prompt_size`: Size of Wikipedia prompts in tokens
- `--num_samples`: Number of Wikipedia samples to process
- `--k_points`: Number of interpolation points between hidden states (default: 10)
- `--top_k_tokens`: Number of top tokens to consider (default: 1)
- `--device`: Device to run on ('auto', 'cpu', 'cuda')

### Output Structure
```
interpolation_experiment_YYYYMMDD_HHMMSS/
├── experiment_summary.json                # High-level statistics
├── detailed_results.json                  # Complete results per sample
├── interpolation_analysis.csv             # Tabular data for analysis
└── interpolation_analysis.png             # Visualization of results
```

### Interpretation
- **Low Unique Token Count** (1-2): Smooth, consistent predictions
- **High Unique Token Count** (5+): Rapid changes, decision boundaries
- **Increasing with Steps**: Accumulating uncertainty in longer generations

---

## 🛠️ Dependencies

All experiments require the following core dependencies (see `requirements.txt`):

### Core Requirements
```bash
torch>=2.0
transformers>=4.30
datasets
numpy>=1.24
scipy>=1.10.0
matplotlib
pandas>=1.5.0
tqdm
```

### Analysis & Visualization
```bash
seaborn
scikit-learn>=1.2
sentence-transformers>=2.2
```

### Experiment-Specific
```bash
# For Experiment 2 (LLM Evaluation)
litellm>=1.0.0
pydantic>=2.0.0
python-dotenv
groq>=0.4.0

# For Advanced Analysis
umap
wandb
tensorboard
h5py
spacy
nltk
```

## 💻 Computational Requirements

### Hardware Recommendations

| Model Size | RAM | VRAM | Estimated Runtime (10 samples) |
|-----------|-----|------|--------------------------------|
| 135M-1B | 4GB+ | 2GB+ | 1-5 minutes |
| 1B-3B | 8GB+ | 6GB+ | 5-15 minutes |
| 7B+ | 16GB+ | 12GB+ | 20+ minutes |

### GPU Acceleration
Most experiments support GPU acceleration. Use `--device cuda` for GPU execution or `--device cpu` for CPU-only execution.

## 📈 Common Usage Patterns

### Quick Testing
All experiments support quick testing with reduced parameters:
```bash
# Minimal parameter sets for testing
--num_samples 5
--num_articles 5
--continuation_size 3
--k_points 5
```

### Batch Processing
Several experiments include batch scripts:
```bash
# Experiment 1: Multiple models
cd experiments/experiment_1_momentum_perplexity_band_trajectory_extractor/
./run_all_models_phase_space.sh

# Experiment 3: Beta fitting batch
cd experiments/experiment_3_momentum_perplexity_variance/
./run_beta_fitting_batch.sh
```

### Custom Output Directories
```bash
# Specify custom output location
--output_dir /path/to/custom/results/
```

## 🔬 Theoretical Background

These experiments are grounded in several theoretical frameworks:

1. **Dynamical Systems Theory**: Language generation as a system with phase space dynamics
2. **Statistical Mechanics**: Thermodynamic approaches to language model behavior
3. **Conservation Laws**: Discovery of invariant quantities in generation processes
4. **Geometric Analysis**: Understanding embedding space structure and decision boundaries

## 🚨 Troubleshooting

### Common Issues

**Memory Errors**:
- Reduce batch sizes or sample counts
- Use smaller models for testing
- Enable CPU-only mode if GPU memory insufficient

**API Errors (Experiment 2)**:
- Verify OpenAI API key is set correctly
- Check API quota and billing status
- Use fallback models if primary API unavailable

**Missing Dependencies**:
```bash
# Install all requirements
pip install -r requirements.txt

# Install specific packages
pip install torch transformers datasets
```

**Model Download Issues**:
- Ensure stable internet connection
- Check HuggingFace model availability
- Use smaller models for testing

### Performance Optimization

**GPU Utilization**:
```bash
# Monitor GPU usage
nvidia-smi

# Use mixed precision for larger models
--device cuda --precision fp16
```

**Memory Management**:
```bash
# Clear cache between runs
python -c "import torch; torch.cuda.empty_cache()"
```

## 📚 Further Reading

- [Phase Space Analysis Theory](experiments/experiment_1_momentum_perplexity_band_trajectory_extractor/README.md)
- [Jacobian Steering Methods](experiments/experiment_2_jacobian_completion_eval/README.md) 
- [Statistical Analysis Techniques](experiments/experiment_3_momentum_perplexity_variance/README.md)
- [Interpolation Studies](experiments/experiment_4_token_transition_interpolation/README.md)

## 🤝 Contributing

When adding new experiments:
1. Create descriptive README.md with usage examples
2. Include example runner scripts
3. Maintain consistent output formats
4. Add appropriate error handling and logging
5. Update this master README with new experiment details

---

**Note**: These experiments require significant computational resources for full analysis. Start with quick test configurations before running full experiments. 