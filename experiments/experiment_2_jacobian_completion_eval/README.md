# 🚀 Experiment 20: Jacobian Steering Completion Evaluation

This experiment evaluates the effectiveness of Jacobian steering by comparing how well natural vs steered token distributions complete sentences from Wikipedia text using LLM-based evaluation.

## 🎯 Objective

Assess whether Jacobian steering produces token distributions that lead to better sentence completions compared to natural generation, as evaluated by an LLM judge.

## 🔬 Methodology

### Overview
1. **Load Wikipedia Data**: Extract articles and create prompt-completion pairs
2. **Generate Distributions**: For each prompt, get natural and Jacobian-steered token distributions
3. **Extract Top-N Tokens**: Get the most probable tokens from both distributions
4. **LLM Evaluation**: Use GPT-4o-mini to score how well each token completes the sentence
5. **Compare Performance**: Analyze average completion quality scores

### Detailed Process

For each Wikipedia text sample:

1. **Prompt Creation**: Use first `prompt_length` tokens as context
2. **Target Token**: Extract the actual next token from Wikipedia as steering target
3. **Natural Generation**: Get token distribution from standard forward pass
4. **Jacobian Steering**: Apply Jacobian path search to steer toward Wikipedia token
5. **Token Extraction**: Get top-N most probable tokens from both distributions
6. **Quality Evaluation**: Use LLM to rate completion quality (1-10 scale) for each token
7. **Score Aggregation**: Calculate weighted and simple averages for each distribution

### Evaluation Criteria

The LLM judge evaluates completions based on:
- **Grammatical Correctness**: Syntactic validity
- **Contextual Appropriateness**: Semantic fit with prompt
- **Naturalness**: Fluency and readability  
- **Factual Accuracy**: Correctness when applicable

## 📊 Metrics

### Primary Metrics
- **Average Completion Score**: Simple mean of top-N token scores
- **Weighted Completion Score**: Probability-weighted mean of top-N token scores
- **Score Improvement**: Difference between steered and natural scores

### Secondary Metrics  
- **Target Token Performance**: Rank and probability of Wikipedia token
- **Steering Effectiveness**: Success rate of achieving top-1 for target
- **Completion Quality Distribution**: Analysis of score patterns

## 🚀 Usage

### Basic Usage
```bash
python run_jacobian_completion_experiment.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --num_samples 50 \
  --prompt_length 20 \
  --continuation_length 1 \
  --top_n 10 \
  --device cuda \
  --output_dir ./results
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | HuggingFace model identifier | `Qwen/Qwen2.5-1.5B-Instruct` |
| `--num_samples` | Number of Wikipedia articles to process | `50` |
| `--prompt_length` | Length of prompt in tokens | `20` |
| `--continuation_length` | Length of continuation in tokens to evaluate | `1` |
| `--top_n` | Number of top tokens to evaluate | `10` |
| `--device` | Computing device (`cuda`, `cpu`, `mps`) | `cpu` |
| `--output_dir` | Results output directory | `./results` |

### Example Commands

**Quick Test**:
```bash
python run_jacobian_completion_experiment.py --num_samples 10 --top_n 5
```

**Multi-token Evaluation**:
```bash
python run_jacobian_completion_experiment.py \
  --num_samples 50 \
  --prompt_length 15 \
  --continuation_length 3 \
  --top_n 10
```

**Full Evaluation**:
```bash
python run_jacobian_completion_experiment.py \
  --model_name microsoft/DialoGPT-medium \
  --num_samples 100 \
  --prompt_length 15 \
  --continuation_length 1 \
  --top_n 15 \
  --device cuda
```

**Large Model**:
```bash
python run_jacobian_completion_experiment.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --num_samples 200 \
  --continuation_length 2 \
  --device cuda
```

## 📁 Output Files

Results are saved to timestamped directories: `{output_dir}/jacobian_completion_{timestamp}/`

### Generated Files

| File | Description |
|------|-------------|
| `completion_evaluation_results.csv` | Main results with scores and metrics |
| `detailed_completion_analysis.json` | Detailed token analysis for first 10 samples |
| `completion_evaluation_analysis.png` | Comprehensive analysis plots |
| `completion_evaluation_summary.png` | Summary statistics visualization |
| `experiment_config.json` | Configuration and metadata |

### CSV Schema

Key columns in `completion_evaluation_results.csv`:

- `sample_id`: Unique identifier
- `article_title`: Wikipedia article title
- `target_text`: Ground truth next token
- `natural_avg_score`: Average LLM score for natural tokens
- `steered_avg_score`: Average LLM score for steered tokens  
- `score_improvement`: Steered - natural average scores
- `target_rank_natural/steered`: Rank of target token in distributions
- `steering_achieved_top1`: Whether steering achieved top-1 for target

## 🔧 Requirements

### Dependencies
```bash
pip install torch transformers datasets pandas matplotlib seaborn tqdm scipy litellm pydantic
```

### Environment Setup
Set OpenAI API key for LLM evaluation:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

### Hardware Requirements
- **GPU**: Recommended for models >1B parameters
- **Memory**: 8GB+ RAM, 6GB+ VRAM for 1.5B models
- **Storage**: 2GB+ for datasets and model weights

## 📊 Analysis

### Statistical Tests
The experiment performs:
- **Paired t-test**: Tests significance of score improvements
- **Wilcoxon signed-rank test**: Non-parametric alternative
- **Correlation analysis**: Relationships between metrics

### Visualizations

1. **Score Comparison Plots**: Natural vs steered performance scatter plots
2. **Improvement Distribution**: Histogram of score changes
3. **Target Rank Analysis**: Effectiveness of steering for target tokens
4. **Success Rate Analysis**: Performance by steering success
5. **Correlation Heatmap**: Relationships between metrics

### Expected Results

**Hypothesis**: Jacobian steering should improve completion quality by:
- Increasing probability of contextually appropriate tokens
- Reducing probability of irrelevant tokens
- Better alignment with ground truth continuations

**Success Criteria**:
- Statistically significant improvement in average scores
- Higher target token ranks in steered distributions
- Positive correlation between steering success and score improvement

## 🧪 Testing

### Quick Test
```bash
python test_experiment.py
```

### Manual Testing
```bash
# Test with minimal configuration
python run_jacobian_completion_experiment.py \
  --num_samples 5 \
  --prompt_length 10 \
  --top_n 3 \
  --device cpu
```

## 🔬 Implementation Details

### Jacobian Steering
- Uses `JacobianPathSearch` from `utils.steering.jacobian_path_search`
- Applies closed-form gradient computation: `∇_h log p_t = W_t - Σ_j p_j W_j`
- Beam search through hidden space with analytical steering directions

### Structured Response Format
The experiment uses Pydantic models for reliable LLM evaluation parsing:

```python
class TokenEvaluation(BaseModel):
    token_text: str      # The token being evaluated
    score: float         # Quality score (1-10)
    reasoning: str       # Brief explanation

class CompletionEvaluationResponse(BaseModel):
    evaluations: List[TokenEvaluation]  # Per-token evaluations
    overall_assessment: str             # General assessment
```

### LLM Evaluation
- Uses `litellm` with GPT-4o-mini for consistency
- Temperature 0.1 for deterministic scoring
- **Structured output** with Pydantic models for reliable parsing
- Fallback to simple JSON format if structured output fails
- Final fallback scores (5.0) for API failures

### Data Processing
- Wikipedia 20220301.en dataset via HuggingFace
- Automatic filtering of short articles
- Multiple samples per article with sliding windows
- Robust error handling for tokenization edge cases

## ⚠️ Limitations

1. **LLM Judge Bias**: GPT-4o-mini may have preferences affecting evaluation
2. **Context Length**: Limited by model's maximum sequence length
3. **Token Granularity**: Single-token evaluation may miss multi-token concepts
4. **Computational Cost**: LLM evaluation adds significant overhead
5. **Language Dependency**: Optimized for English Wikipedia content

## 🔄 Future Extensions

1. **Multi-token Evaluation**: Evaluate completion sequences rather than single tokens
2. **Domain Variation**: Test on scientific, literary, or technical texts
3. **Model Comparison**: Systematic evaluation across model architectures
4. **Human Evaluation**: Validate LLM judge against human annotators
5. **Semantic Coherence**: Integrate semantic similarity metrics
6. **Dynamic Scoring**: Adapt evaluation criteria based on context type

## 📚 References

- [Jacobian Path Search Steering](../../utils/steering/jacobian_path_search.py)
- [Wikipedia Dataset](https://huggingface.co/datasets/wikipedia)
- [LiteLLM Documentation](https://docs.litellm.ai/)

## 🤝 Contributing

When modifying this experiment:

1. **Maintain Compatibility**: Preserve output format for downstream analysis
2. **Document Changes**: Update README and docstrings
3. **Test Thoroughly**: Validate on small samples before full runs
4. **Version Control**: Tag significant changes for reproducibility

---

**Note**: This experiment requires OpenAI API access for LLM evaluation. Ensure proper API key configuration and monitor usage costs. 