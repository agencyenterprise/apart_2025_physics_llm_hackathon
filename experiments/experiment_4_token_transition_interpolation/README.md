# Experiment 23: Token Transition Interpolation

## Overview

This experiment analyzes how token predictions change along interpolated paths between consecutive hidden states during text generation. The key insight is to understand the "smoothness" of the language model's decision space by examining how many different top tokens appear when we linearly interpolate between consecutive hidden states.

## Methodology

For each generated token in a continuation sequence, the experiment:

1. **Captures Hidden States**: Records the hidden state before and after generating each token
2. **Creates Interpolation Path**: Generates `k` linear interpolation points between consecutive hidden states
3. **Extracts Token Predictions**: Applies the language model head (lm_head + softmax) to each interpolation point
4. **Counts Unique Tokens**: Measures how many different top tokens appear across the interpolation path

### Mathematical Framework

Given two consecutive hidden states `h₁` and `h₂`, we create interpolation points:
```
h_α = (1 - α) * h₁ + α * h₂
```
where `α ∈ [0, 1]` with `k` equally spaced values.

For each interpolated hidden state `h_α`, we compute:
```
logits_α = lm_head(h_α)
probs_α = softmax(logits_α)
top_tokens_α = argmax_k(probs_α)
```

The main metric is the cardinality of the union of all top tokens across the interpolation path.

## Key Research Questions

- **Decision Boundary Smoothness**: How smoothly do token predictions change in hidden state space?
- **Model Consistency**: Are there abrupt transitions in token preferences along interpolation paths?
- **Architectural Differences**: Do different model architectures show different interpolation behaviors?
- **Generation Context**: How does the interpolation behavior change across different generation steps?

## Usage

### Basic Usage

```bash
python token_transition_interpolation_experiment.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --continuation_size 10 \
    --prompt_size 50 \
    --num_samples 100 \
    --k_points 10
```

### Parameters

- `--model`: HuggingFace model identifier
- `--continuation_size`: Number of tokens to generate per sample
- `--prompt_size`: Size of Wikipedia prompts in tokens (None = use full text)
- `--num_samples`: Number of Wikipedia samples to process
- `--k_points`: Number of interpolation points between hidden states (default: 10)
- `--top_k_tokens`: Number of top tokens to consider for uniqueness counting (default: 1)
- `--device`: Device to run on ('auto', 'cpu', 'cuda')
- `--output_dir`: Custom output directory name

### Example Configurations

#### Quick Test (Small Model)
```bash
python token_transition_interpolation_experiment.py \
    --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --continuation_size 5 \
    --prompt_size 30 \
    --num_samples 10 \
    --k_points 5
```

#### Full Analysis (Large Model)
```bash
python token_transition_interpolation_experiment.py \
    --model "meta-llama/Llama-3.2-3B-Instruct" \
    --continuation_size 15 \
    --prompt_size 100 \
    --num_samples 200 \
    --k_points 20 \
    --top_k_tokens 3
```

### Running Examples

Use the provided runner script to test multiple configurations:
```bash
python run_example.py
```

## Output Structure

The experiment creates a timestamped output directory containing:

### Files Generated

1. **`experiment_summary.json`**: High-level statistics and configuration
2. **`detailed_results.json`**: Complete results for each sample and interpolation
3. **`interpolation_analysis.csv`**: Tabular data for easy analysis
4. **`interpolation_analysis.png`**: Visualization of results

### Key Metrics

- **Mean Unique Tokens per Interpolation**: Average number of unique tokens across interpolation paths
- **Unique Token Distribution**: Statistical distribution of unique token counts
- **Total Unique Tokens**: Total vocabulary diversity observed across all interpolations
- **Step-wise Analysis**: How interpolation behavior changes across generation steps

## Interpretation

### Expected Results

- **Low Unique Token Count** (1-2): Indicates smooth, consistent token predictions across interpolation paths
- **High Unique Token Count** (5+): Suggests rapid changes in token preferences, indicating decision boundaries
- **Increasing with Generation Step**: May indicate accumulating uncertainty in longer generations

### Model Comparisons

Different model architectures may show:
- **Transformer variants**: Different interpolation smoothness based on attention mechanisms
- **Model size effects**: Larger models may show smoother interpolations
- **Training differences**: Instruction-tuned vs base models may behave differently

## Computational Requirements

### Memory Usage
- Depends on model size and hidden dimension
- Stores `k_points × hidden_dim` tensors per interpolation
- GPU memory scales with batch size and model parameters

### Runtime Estimates
- Small models (135M-1B): ~1-5 minutes per 10 samples
- Medium models (1B-3B): ~5-15 minutes per 10 samples  
- Large models (7B+): ~20+ minutes per 10 samples

## Dependencies

Required packages (see main `requirements.txt`):
- `torch`
- `transformers`
- `datasets`
- `numpy`
- `matplotlib`
- `seaborn`
- `pandas`
- `tqdm`

## Theoretical Background

This experiment is related to:
- **Activation Space Geometry**: Understanding the structure of neural network representations
- **Decision Boundary Analysis**: Examining how models make discrete choices from continuous representations
- **Interpolation Studies**: Related to work on latent space interpolation in generative models
- **Linguistic Gradience**: How continuous mathematical representations map to discrete linguistic units

## Extensions

Potential modifications and extensions:
1. **Non-linear Interpolation**: Use spherical or geodesic interpolation instead of linear
2. **Multi-step Paths**: Analyze longer interpolation sequences across multiple generation steps
3. **Attention Visualization**: Examine how attention patterns change along interpolation paths
4. **Layer-wise Analysis**: Compare interpolation behavior across different transformer layers
5. **Semantic Analysis**: Analyze semantic coherence of tokens along interpolation paths

## References

This experiment builds on concepts from:
- Neural network interpretability research
- Continuous relaxations of discrete language models
- Geometric analysis of embedding spaces
- Studies of decision boundaries in neural networks 