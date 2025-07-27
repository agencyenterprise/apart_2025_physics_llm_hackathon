#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token Transition Interpolation Experiment

This experiment analyzes how token predictions change along interpolated paths 
between consecutive hidden states during text generation. For each pair of 
consecutive hidden states, we create k interpolation points and measure how 
many different top tokens appear along the interpolation path.

Usage:
    python token_transition_interpolation_experiment.py --model "Qwen/Qwen2.5-1.5B-Instruct" --continuation_size 10 --prompt_size 50 --num_samples 100 --k_points 10
    python token_transition_interpolation_experiment.py --model "meta-llama/Llama-3.2-1B" --continuation_size 5 --prompt_size 30 --num_samples 50 --k_points 5
"""
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.model_utils import load_random_weight_model

# Set matplotlib backend and font settings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.rcParams['font.size'] = 10


class TokenTransitionInterpolationExperiment:
    """
    Analyzes token prediction changes along interpolated hidden state paths.
    
    For each generated token, this experiment:
    1. Takes the previous hidden state and current hidden state
    2. Creates k interpolation points along the line between them
    3. Applies lm_head + softmax to each point to get token predictions
    4. Counts how many unique top tokens appear across the interpolation path
    """
    
    def __init__(self, 
                 model_name: str,
                 device: str = "auto",
                 k_points: int = 10,
                 top_k_tokens: int = 1):
        """
        Initialize the experiment.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("auto", "cpu", "cuda")
            k_points: Number of interpolation points between hidden states
            top_k_tokens: Number of top tokens to consider for uniqueness counting
        """
        self.model_name = model_name
        self.k_points = k_points
        self.top_k_tokens = top_k_tokens
        
        # Setup device
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Token Transition Interpolation Experiment")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Interpolation points (k): {k_points}")
        print(f"   Top-k tokens for analysis: {top_k_tokens}")
        
        # Load model and tokenizer
        print(f"📦 Loading model and tokenizer...")
        
        
        
        if model_name.endswith(":random"):
            print(f"🎲 Loading random-weight baseline for: {model_name}")
            base_model_name = model_name[:-7]  # Remove ":random" suffix
            print(f"🎲 Loading random-weight baseline for: {base_model_name}")
            self.model, self.tokenizer, self.config = load_random_weight_model(
                model_name=base_model_name,
                device=device,
                torch_dtype=torch.float32
            )
            self.model_name = base_model_name
            self.is_random_model = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for numerical stability
                trust_remote_code=True,
                device_map=None
            ).to(self.device)
            self.model_name = model_name
            self.is_random_model = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        print(f"✅ Model loaded successfully")
        print(f"   Hidden dimension: {self.model.config.hidden_size}")
        print(f"   Vocabulary size: {self.model.config.vocab_size}")
    
    def load_wikipedia_data(self, num_samples: int, min_length: int = 200) -> List[str]:
        """
        Load Wikipedia data for prompts.
        
        Args:
            num_samples: Number of Wikipedia articles to load
            min_length: Minimum text length for articles
            
        Returns:
            List of Wikipedia text samples
        """
        print(f"📚 Loading {num_samples} Wikipedia articles...")
        
        try:
            # Load Wikipedia dataset
            dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
            
            texts = []
            for i, example in enumerate(dataset):
                if len(texts) >= num_samples:
                    break
                
                text = example["text"]
                if len(text) >= min_length:
                    # Take first part as sample
                    sentences = text.split('. ')
                    if len(sentences) >= 3:
                        sample_text = '. '.join(sentences[:4]) + '.'
                        texts.append(sample_text)
                
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1} articles, collected {len(texts)} valid samples")
            
            print(f"✅ Loaded {len(texts)} Wikipedia samples")
            return texts
            
        except Exception as e:
            print(f"❌ Failed to load Wikipedia data: {e}")
            # Fallback to sample texts
            print("🔄 Using fallback sample texts")
            return [
                "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space and time. It consists of two related theories: special relativity and general relativity.",
                "Machine learning algorithms have become increasingly sophisticated in recent years. They can now perform tasks that were once thought to be exclusively human.",
                "The human brain contains approximately 86 billion neurons that communicate through synapses. This complex network enables consciousness and cognition.",
                "Climate change is one of the most pressing global challenges of the 21st century. It affects weather patterns, sea levels, and ecosystems worldwide.",
                "The invention of the printing press by Johannes Gutenberg in the 15th century transformed information sharing. It made books more accessible to the general public."
            ][:num_samples]
    
    def interpolate_hidden_states(self, 
                                hidden_state_1: torch.Tensor, 
                                hidden_state_2: torch.Tensor, 
                                num_points: int) -> List[torch.Tensor]:
        """
        Create linear interpolation points between two hidden states.
        
        Args:
            hidden_state_1: Starting hidden state [hidden_dim]
            hidden_state_2: Ending hidden state [hidden_dim]
            num_points: Number of interpolation points (including endpoints)
            
        Returns:
            List of interpolated hidden states
        """
        # Create interpolation weights
        alphas = torch.linspace(0, 1, num_points, device=self.device)
        
        interpolated_states = []
        for alpha in alphas:
            # Linear interpolation: (1-α) * h1 + α * h2
            interpolated = (1 - alpha) * hidden_state_1 + alpha * hidden_state_2
            interpolated_states.append(interpolated)
        
        return interpolated_states
    
    def get_top_tokens_from_hidden_state(self, hidden_state: torch.Tensor) -> Tuple[List[int], List[float]]:
        """
        Get top-k token predictions from a hidden state.
        
        Args:
            hidden_state: Hidden state tensor [hidden_dim]
            
        Returns:
            Tuple of (top_token_ids, top_probabilities)
        """
        with torch.no_grad():
            # Apply language model head
            logits = self.model.lm_head(hidden_state)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k tokens
            top_probs, top_tokens = torch.topk(probs, self.top_k_tokens)
            
            return top_tokens.cpu().tolist(), top_probs.cpu().tolist()
    
    def analyze_interpolation_path(self, 
                                 hidden_state_1: torch.Tensor, 
                                 hidden_state_2: torch.Tensor) -> Dict:
        """
        Analyze token predictions along an interpolation path.
        
        Args:
            hidden_state_1: Starting hidden state
            hidden_state_2: Ending hidden state
            
        Returns:
            Dictionary with analysis results
        """
        # Create interpolation points
        interpolated_states = self.interpolate_hidden_states(
            hidden_state_1, hidden_state_2, self.k_points
        )
        
        # Get token predictions for each point
        all_tokens = set()
        path_predictions = []
        
        for i, state in enumerate(interpolated_states):
            top_tokens, top_probs = self.get_top_tokens_from_hidden_state(state)
            
            # Add to unique token set
            all_tokens.update(top_tokens)
            
            # Store predictions for this point
            path_predictions.append({
                'interpolation_step': i,
                'alpha': i / (self.k_points - 1),
                'top_tokens': top_tokens,
                'top_probabilities': top_probs,
                'top_token_strings': [self.tokenizer.decode([t]) for t in top_tokens]
            })
        
        return {
            'unique_token_count': len(all_tokens),
            'unique_tokens': list(all_tokens),
            'path_predictions': path_predictions,
            'total_interpolation_points': self.k_points
        }
    
    def generate_with_hidden_state_tracking(self, 
                                          text: str, 
                                          prompt_size: Optional[int], 
                                          continuation_size: int) -> Dict:
        """
        Generate text while tracking hidden states for interpolation analysis.
        
        Args:
            text: Input text
            prompt_size: Size of prompt in tokens (None = use all text)
            continuation_size: Number of tokens to generate
            
        Returns:
            Dictionary with generation results and analysis
        """
        # Tokenize input
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Limit prompt size if specified
        if prompt_size is not None and len(tokens) > prompt_size:
            tokens = tokens[:prompt_size]
        
        # Convert to tensor
        current_ids = torch.tensor([tokens], device=self.device)
        
        print(f"   Prompt length: {len(tokens)} tokens")
        print(f"   Generating {continuation_size} tokens...")
        
        # Track hidden states and analyze interpolations
        hidden_states_history = []
        interpolation_analyses = []
        generated_tokens = []
        
        with torch.no_grad():
            for step in range(continuation_size):
                # Forward pass to get hidden states
                outputs = self.model(current_ids, output_hidden_states=True)
                
                # Get the final hidden state (last layer, last token)
                current_hidden = outputs.hidden_states[-1][0, -1, :]  # [hidden_dim]
                hidden_states_history.append(current_hidden.clone())
                
                # Generate next token
                next_token_logits = self.model.lm_head(current_hidden)
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(next_token_probs).item()
                
                generated_tokens.append(next_token)
                
                # Add token to sequence
                next_token_tensor = torch.tensor([[next_token]], device=self.device)
                current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
                
                # Analyze interpolation path (if we have a previous hidden state)
                if len(hidden_states_history) >= 2:
                    prev_hidden = hidden_states_history[-2]
                    curr_hidden = hidden_states_history[-1]
                    
                    analysis = self.analyze_interpolation_path(prev_hidden, curr_hidden)
                    analysis['step'] = step
                    analysis['generated_token'] = next_token
                    analysis['generated_token_string'] = self.tokenizer.decode([next_token])
                    
                    interpolation_analyses.append(analysis)
                
                print(f"     Step {step + 1}: Generated '{self.tokenizer.decode([next_token])}'")
        
        # Compile results
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return {
            'prompt_tokens': tokens,
            'generated_tokens': generated_tokens,
            'generated_text': generated_text,
            'full_text': text + generated_text,
            'hidden_states_history': [h.cpu().numpy() for h in hidden_states_history],
            'interpolation_analyses': interpolation_analyses,
            'total_interpolations': len(interpolation_analyses)
        }
    
    def run_experiment(self, 
                      prompt_size: Optional[int],
                      continuation_size: int,
                      num_samples: int,
                      output_dir: str = None) -> Dict:
        """
        Run the complete experiment.
        
        Args:
            prompt_size: Size of Wikipedia prompts in tokens
            continuation_size: Number of tokens to generate per sample
            num_samples: Number of samples to process
            output_dir: Directory to save results
            
        Returns:
            Dictionary with experiment results
        """
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"token_transition_interpolation_{timestamp}"
        
        output_path = Path(f"./output_dir")
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Results will be saved to: {output_path}")
        
        # Load Wikipedia data
        wikipedia_texts = self.load_wikipedia_data(num_samples)
        
        # Run analysis on each sample
        results = []
        all_interpolation_analyses = []
        
        print(f"\n🔬 Running experiment on {len(wikipedia_texts)} samples...")
        
        for i, text in enumerate(tqdm(wikipedia_texts, desc="Processing samples")):
            print(f"\n📝 Sample {i + 1}/{len(wikipedia_texts)}")
            
            try:
                sample_result = self.generate_with_hidden_state_tracking(
                    text, prompt_size, continuation_size
                )
                
                sample_result['sample_id'] = i
                sample_result['source_text'] = text
                results.append(sample_result)
                
                # Collect interpolation analyses
                all_interpolation_analyses.extend(sample_result['interpolation_analyses'])
                
            except Exception as e:
                print(f"⚠️  Error processing sample {i + 1}: {e}")
                continue
        
        # Compile experiment summary
        experiment_summary = self.compile_experiment_summary(results, all_interpolation_analyses)
        
        # Save results
        self.save_results(results, experiment_summary, output_path)
        
        # Create visualizations
        self.create_visualizations(experiment_summary, output_path)
        
        print(f"\n✅ Experiment completed!")
        print(f"   Processed {len(results)} samples")
        print(f"   Total interpolation analyses: {len(all_interpolation_analyses)}")
        print(f"   Results saved to: {output_path}")
        
        return experiment_summary
    
    def compile_experiment_summary(self, results: List[Dict], all_analyses: List[Dict]) -> Dict:
        """Compile summary statistics from experiment results."""
        
        # Extract unique token counts
        unique_token_counts = [analysis['unique_token_count'] for analysis in all_analyses]
        
        # Extract interpolation statistics
        total_unique_tokens = set()
        for analysis in all_analyses:
            total_unique_tokens.update(analysis['unique_tokens'])
        
        # Compile statistics
        summary = {
            'experiment_config': {
                'model_name': self.model_name,
                'k_points': self.k_points,
                'top_k_tokens': self.top_k_tokens,
                'total_samples': len(results),
                'total_interpolations': len(all_analyses)
            },
            'unique_token_statistics': {
                'mean_unique_tokens_per_interpolation': np.mean(unique_token_counts),
                'std_unique_tokens_per_interpolation': np.std(unique_token_counts),
                'min_unique_tokens': np.min(unique_token_counts),
                'max_unique_tokens': np.max(unique_token_counts),
                'median_unique_tokens': np.median(unique_token_counts),
                'total_unique_tokens_seen': len(total_unique_tokens)
            },
            'distribution_data': {
                'unique_token_counts': unique_token_counts,
                'all_analyses': all_analyses
            }
        }
        
        return summary
    
    def save_results(self, results: List[Dict], summary: Dict, output_path: Path):
        """Save experiment results to files."""
        
        # Save summary
        summary_file = output_path / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_summary = self.convert_numpy_types(summary)
            json.dump(json_summary, f, indent=2)
        
        # Save detailed results
        results_file = output_path / "detailed_results.json"
        with open(results_file, 'w') as f:
            json_results = self.convert_numpy_types(results)
            json.dump(json_results, f, indent=2)
        
        # Save analysis data as CSV for easy analysis
        csv_data = []
        for result in results:
            for analysis in result['interpolation_analyses']:
                csv_data.append({
                    'sample_id': result['sample_id'],
                    'step': analysis['step'],
                    'unique_token_count': analysis['unique_token_count'],
                    'generated_token': analysis['generated_token_string'],
                    'interpolation_points': analysis['total_interpolation_points']
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = output_path / "interpolation_analysis.csv"
            df.to_csv(csv_file, index=False)
        
        print(f"💾 Results saved:")
        print(f"   Summary: {summary_file}")
        print(f"   Detailed: {results_file}")
        print(f"   CSV: {csv_file}")
    
    def create_visualizations(self, summary: Dict, output_path: Path):
        """Create visualizations of the experiment results."""
        
        unique_token_counts = summary['distribution_data']['unique_token_counts']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Token Transition Interpolation Analysis\nModel: {self.model_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Distribution of unique token counts
        axes[0, 0].hist(unique_token_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Unique Tokens per Interpolation Path')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Distribution of Unique Token Counts\n(k={self.k_points} interpolation points)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Mean: {summary['unique_token_statistics']['mean_unique_tokens_per_interpolation']:.2f}
Std: {summary['unique_token_statistics']['std_unique_tokens_per_interpolation']:.2f}
Median: {summary['unique_token_statistics']['median_unique_tokens']:.2f}"""
        axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Box plot of unique token counts
        axes[0, 1].boxplot(unique_token_counts, patch_artist=True, 
                          boxprops=dict(facecolor='lightcoral', alpha=0.7))
        axes[0, 1].set_ylabel('Unique Tokens per Interpolation Path')
        axes[0, 1].set_title('Box Plot of Unique Token Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Unique tokens vs interpolation step (if we have step data)
        all_analyses = summary['distribution_data']['all_analyses']
        if all_analyses:
            steps = [analysis['step'] for analysis in all_analyses]
            unique_counts = [analysis['unique_token_count'] for analysis in all_analyses]
            
            # Create scatter plot
            axes[1, 0].scatter(steps, unique_counts, alpha=0.6, color='green')
            axes[1, 0].set_xlabel('Generation Step')
            axes[1, 0].set_ylabel('Unique Tokens in Interpolation Path')
            axes[1, 0].set_title('Unique Tokens vs Generation Step')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary statistics table
        axes[1, 1].axis('off')
        stats_table = [
            ['Metric', 'Value'],
            ['Total Samples', f"{summary['experiment_config']['total_samples']}"],
            ['Total Interpolations', f"{summary['experiment_config']['total_interpolations']}"],
            ['K Points', f"{summary['experiment_config']['k_points']}"],
            ['Top-K Tokens', f"{summary['experiment_config']['top_k_tokens']}"],
            ['Mean Unique Tokens', f"{summary['unique_token_statistics']['mean_unique_tokens_per_interpolation']:.2f}"],
            ['Max Unique Tokens', f"{summary['unique_token_statistics']['max_unique_tokens']}"],
            ['Total Unique Tokens', f"{summary['unique_token_statistics']['total_unique_tokens_seen']}"]
        ]
        
        table = axes[1, 1].table(cellText=stats_table[1:], colLabels=stats_table[0],
                                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        axes[1, 1].set_title('Experiment Summary', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_path / "interpolation_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved: {plot_file}")
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Token Transition Interpolation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python token_transition_interpolation_experiment.py --model "Qwen/Qwen2.5-1.5B-Instruct" --continuation_size 10 --prompt_size 50 --num_samples 100 --k_points 10
  python token_transition_interpolation_experiment.py --model "meta-llama/Llama-3.2-1B" --continuation_size 5 --prompt_size 30 --num_samples 50 --k_points 5
        """
    )
    
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="HuggingFace model identifier (e.g., 'Qwen/Qwen2.5-1.5B-Instruct')")
    parser.add_argument("--continuation_size", type=int, default=10,
                       help="Number of tokens to generate per sample")
    parser.add_argument("--prompt_size", type=int, default=None,
                       help="Size of Wikipedia prompts in tokens (None = use full text)")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of Wikipedia samples to process")
    parser.add_argument("--k_points", type=int, default=10,
                       help="Number of interpolation points between hidden states (default: 10)")
    parser.add_argument("--top_k_tokens", type=int, default=1,
                       help="Number of top tokens to consider for uniqueness counting (default: 1)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run on: 'auto', 'cpu', 'cuda' (default: auto)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (default: auto-generated)")
    
    args = parser.parse_args()
    
    print("🚀 Starting Token Transition Interpolation Experiment")
    print(f"Arguments: {vars(args)}")
    
    # Create experiment instance
    experiment = TokenTransitionInterpolationExperiment(
        model_name=args.model,
        device=args.device,
        k_points=args.k_points,
        top_k_tokens=args.top_k_tokens
    )
    
    # Run experiment
    results = experiment.run_experiment(
        prompt_size=args.prompt_size,
        continuation_size=args.continuation_size,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    print("\n🎉 Experiment completed successfully!")
    print(f"📊 Key Results:")
    print(f"   Mean unique tokens per interpolation: {results['unique_token_statistics']['mean_unique_tokens_per_interpolation']:.2f}")
    print(f"   Total unique tokens observed: {results['unique_token_statistics']['total_unique_tokens_seen']}")
    print(f"   Max unique tokens in single path: {results['unique_token_statistics']['max_unique_tokens']}")


if __name__ == "__main__":
    main() 