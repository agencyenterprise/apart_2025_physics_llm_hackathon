#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Experiment 20: Jacobian Steering Completion Evaluation with Embedding Baselines

This experiment evaluates the quality of Jacobian steering by comparing sequential
generation between four different approaches using Wikipedia text:

1. Natural generation (greedy decoding with lm_head)
2. Jacobian-steered generation (steering toward ground truth token at each step with lm_head)
3. Natural embedding baseline (greedy decoding with embedding similarity)
4. Steered embedding baseline (steering toward ground truth token with embedding similarity)

For each Wikipedia sample:
1. Extract a prompt from Wikipedia text (length: prompt_length tokens)
2. Generate continuation_length tokens sequentially using all four approaches
3. Use LLM to evaluate the quality of complete generated continuations
4. Analyze steering effectiveness, sequence divergence, and embedding vs lm_head comparison
5. Generate comprehensive statistical reports with p-values and effect sizes

Usage:
python run_jacobian_completion_experiment.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --num_samples 50 \
  --prompt_length 20 \
  --continuation_length 5 \
  --device cuda \
  --output_dir ./results
"""

import sys
import argparse
import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any
import warnings
from dotenv import load_dotenv
load_dotenv()
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import litellm
from tqdm import tqdm
import numpy as np
from scipy import stats
from pydantic import BaseModel, Field

# Add utils to path
sys.path.append("../../")
from utils.steering.jacobian_path_search import JacobianPathSearch

warnings.filterwarnings('ignore')


class TokenEvaluation(BaseModel):
    """Evaluation of a single token completion."""
    token_text: str = Field(..., description="The token being evaluated")
    score: float = Field(..., ge=1.0, le=10.0, description="Completion quality score from 1-10")
    reasoning: str = Field(..., description="Brief explanation for the score")


class CompletionEvaluationResponse(BaseModel):
    """Structured response for completion evaluation."""
    evaluations: List[TokenEvaluation] = Field(..., description="List of token evaluations")
    overall_assessment: str = Field(..., description="Overall assessment of the completion options")

class JacobianCompletionExperiment:
    """Evaluates Jacobian steering quality using LLM-based completion scoring."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cuda"):
        """Initialize the experiment with model and tokenizer."""
        self.device = torch.device(device)
        print(f"🔧 Using device: {self.device}")
        
        # Load model and tokenizer
        print(f"📦 Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            output_hidden_states=True
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Initialize Jacobian steering
        self.jacobian_steering = JacobianPathSearch(self.model, self.tokenizer, self.device)
        
        # Extract embedding matrix for baselines
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            # For models like Llama, Qwen
            self.embedding_matrix = self.model.model.embed_tokens.weight.data
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            # For GPT-like models
            self.embedding_matrix = self.model.transformer.wte.weight.data
        elif hasattr(self.model, 'embeddings'):
            # For BERT-like models
            self.embedding_matrix = self.model.embeddings.word_embeddings.weight.data
        else:
            # Fallback - try to find embedding layer
            for name, module in self.model.named_modules():
                if 'embed' in name.lower() and hasattr(module, 'weight'):
                    self.embedding_matrix = module.weight.data
                    break
            else:
                raise ValueError("Could not find embedding matrix in model")
        
        print(f"✅ Model loaded successfully")
        print(f"   Vocab size: {self.model.config.vocab_size}")
        print(f"   Hidden size: {self.model.config.hidden_size}")
        print(f"   Embedding matrix shape: {self.embedding_matrix.shape}")
    
    def load_wikipedia_data(self, num_samples: int = 50) -> List[Dict]:
        """Load Wikipedia articles for testing."""
        print(f"📚 Loading {num_samples} Wikipedia articles...")
        
        ds = load_dataset("wikipedia", "20220301.en", split="train").shuffle(seed=42)
        articles = ds.select(range(num_samples * 2))  # Get extra in case some are too short
        
        wiki_data = []
        for i, rec in enumerate(articles):
            if len(wiki_data) >= num_samples:
                break
                
            # Tokenize article
            tokens = self.tokenizer(
                rec["text"], 
                return_tensors="pt", 
                truncation=True,
                max_length=512
            ).input_ids[0]  # Remove batch dimension
            
            # Skip if too short
            if len(tokens) < 50:
                continue
                
            wiki_data.append({
                "tokens": tokens,
                "text": rec["text"],
                "title": rec.get("title", f"Article_{i}")
            })
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1} articles, kept {len(wiki_data)} valid ones")
        
        print(f"✅ Loaded {len(wiki_data)} valid Wikipedia articles")
        return wiki_data
    
    def extract_completion_samples(self, wiki_data: List[Dict], prompt_length: int = 20, continuation_length: int = 1) -> List[Dict]:
        """Extract prompts with ground truth continuations from Wikipedia data."""
        print(f"🔄 Extracting prompts (length {prompt_length}) + ground truth continuations (length {continuation_length})...")
        
        completion_samples = []
        sample_id = 0
        
        for article in wiki_data:
            tokens = article["tokens"]
            text = article["text"]
            title = article["title"]
            
            # Extract one prompt + continuation per article  
            if len(tokens) >= prompt_length + continuation_length + 10:  # Need buffer
                # Take prompt from somewhere in the middle of the article
                start_idx = 0  # Start at 25% through the article
                prompt_tokens = tokens[start_idx:start_idx + prompt_length]
                target_continuation_tokens = tokens[start_idx + prompt_length:start_idx + prompt_length + continuation_length]
                
                # Convert to text
                prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                target_continuation_text = self.tokenizer.decode(target_continuation_tokens, skip_special_tokens=True)
                
                # Skip if prompt is too short or continuation is empty
                if len(prompt_text.strip()) >= 20 and len(target_continuation_text.strip()) > 0:
                    completion_samples.append({
                        "sample_id": sample_id,
                        "article_title": title,
                        "prompt_tokens": prompt_tokens,
                        "prompt_text": prompt_text,
                        "target_continuation_tokens": target_continuation_tokens,
                        "target_continuation_text": target_continuation_text,
                        "continuation_length": continuation_length
                    })
                    
                    sample_id += 1
        
        print(f"✅ Extracted {len(completion_samples)} prompt + ground truth continuation pairs")
        return completion_samples
    
    def compute_embedding_based_distribution(self, token_id: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute token distribution based on embedding similarity using a reference token.
        
        Args:
            token_id: Reference token ID to use for similarity computation
            temperature: Temperature for softmax
            
        Returns:
            Token probabilities [vocab_size]
        """
        # Get the embedding vector for the reference token
        reference_embedding = self.embedding_matrix[token_id]  # [hidden_dim]
        
        # Normalize vectors for cosine similarity
        ref_norm = F.normalize(reference_embedding.unsqueeze(0), dim=1)  # [1, hidden_dim]
        embed_norm = F.normalize(self.embedding_matrix, dim=1)  # [vocab_size, hidden_dim]
        
        # Compute cosine similarities between reference token and all embeddings
        similarities = torch.mm(ref_norm, embed_norm.t()).squeeze(0)  # [vocab_size]
        
        # Apply temperature and convert to probabilities
        logits = similarities / temperature
        probs = F.softmax(logits, dim=0)
        
        return probs
    
    def _make_json_serializable(self, obj):
        """Recursively convert objects to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            else:
                return obj.detach().cpu().tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def generate_sequences(self, prompt_tokens: torch.Tensor, target_continuation_tokens: List[int]) -> Dict[str, Any]:
        """Generate natural, steered, and embedding baseline sequences using ground truth targets for steering."""
        
        continuation_length = len(target_continuation_tokens)
        
        # Add batch dimension and move to device
        initial_prompt = prompt_tokens.unsqueeze(0).to(self.device)
        
        # Initialize sequences
        natural_sequence = initial_prompt.clone()
        steered_sequence = initial_prompt.clone()
        natural_embedding_sequence = initial_prompt.clone()
        steered_embedding_sequence = initial_prompt.clone()
        
        # Track generation info for each step
        generation_steps = []
        
        with torch.no_grad():
            for step in range(continuation_length):
                print(f"     🔄 Generation step {step + 1}/{continuation_length}")
                
                # Get natural distribution
                natural_outputs = self.model(natural_sequence)
                natural_logits = natural_outputs.logits[0, -1]  # Last token logits
                natural_probs = F.softmax(natural_logits, dim=-1)
                natural_hidden = natural_outputs.hidden_states[-1][0, -1]  # Last layer, last token
                
                # Select next token for natural generation (greedy)
                natural_next_token = torch.argmax(natural_probs).unsqueeze(0).unsqueeze(0)
                
                # Get steered distribution (steer toward ground truth Wikipedia token)
                target_token_id = target_continuation_tokens[step]
                steering_result = self.jacobian_steering.steer(
                    baseline_hidden=natural_hidden,
                    target_token_id=target_token_id,
                    baseline_probs=natural_probs,
                    debug=False
                )
                
                # Get steered distribution
                steered_hidden = steering_result['final_hidden']
                steered_logits = self.model.lm_head(steered_hidden)
                steered_probs = F.softmax(steered_logits, dim=-1)
                
                # Select next token for steered generation (greedy)
                steered_next_token = torch.argmax(steered_probs).unsqueeze(0).unsqueeze(0)
                
                # Get embedding-based distributions as baselines
                # Use the top tokens from lm_head as reference points for embedding similarity
                natural_embedding_probs = self.compute_embedding_based_distribution(natural_next_token.item())
                steered_embedding_probs = self.compute_embedding_based_distribution(steered_next_token.item())
                
                # Select next tokens for embedding baselines (greedy)
                natural_embedding_next_token = torch.argmax(natural_embedding_probs).unsqueeze(0).unsqueeze(0)
                steered_embedding_next_token = torch.argmax(steered_embedding_probs).unsqueeze(0).unsqueeze(0)
                
                # Store step information
                step_info = {
                    "step": step,
                    "natural_probs": natural_probs.detach().cpu(),
                    "steered_probs": steered_probs.detach().cpu(),
                    "natural_embedding_probs": natural_embedding_probs.detach().cpu(),
                    "steered_embedding_probs": steered_embedding_probs.detach().cpu(),
                    "natural_token": natural_next_token.item(),
                    "steered_token": steered_next_token.item(),
                    "natural_embedding_token": natural_embedding_next_token.item(),
                    "steered_embedding_token": steered_embedding_next_token.item(),
                    "target_token": target_token_id,
                    "natural_token_text": self.tokenizer.decode([natural_next_token.item()]),
                    "steered_token_text": self.tokenizer.decode([steered_next_token.item()]),
                    "natural_embedding_token_text": self.tokenizer.decode([natural_embedding_next_token.item()]),
                    "steered_embedding_token_text": self.tokenizer.decode([steered_embedding_next_token.item()]),
                    "target_token_text": self.tokenizer.decode([target_token_id]),
                    "steering_result": {k: (v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v) 
                                     for k, v in steering_result.items() if not isinstance(v, torch.Tensor) or v.numel() == 1}
                }
                generation_steps.append(step_info)
                
                # Update sequences
                natural_sequence = torch.cat([natural_sequence, natural_next_token], dim=1)
                steered_sequence = torch.cat([steered_sequence, steered_next_token], dim=1)
                natural_embedding_sequence = torch.cat([natural_embedding_sequence, natural_embedding_next_token], dim=1)
                steered_embedding_sequence = torch.cat([steered_embedding_sequence, steered_embedding_next_token], dim=1)
                
                # Debug output
                if step < 3:  # Only show first few steps
                    print(f"       Target:           '{step_info['target_token_text']}' (Ground truth from Wikipedia)")
                    print(f"       Natural:          '{step_info['natural_token_text']}' (prob: {natural_probs[natural_next_token.item()].item():.4f})")
                    print(f"       Steered:          '{step_info['steered_token_text']}' (prob: {steered_probs[steered_next_token.item()].item():.4f})")
                    print(f"       Natural Embed:    '{step_info['natural_embedding_token_text']}' (prob: {natural_embedding_probs[natural_embedding_next_token.item()].item():.4f})")
                    print(f"       Steered Embed:    '{step_info['steered_embedding_token_text']}' (prob: {steered_embedding_probs[steered_embedding_next_token.item()].item():.4f})")
        
        # Extract generated portions (excluding initial prompt)
        prompt_length = initial_prompt.shape[1]
        natural_generated = natural_sequence[0, prompt_length:]
        steered_generated = steered_sequence[0, prompt_length:]
        natural_embedding_generated = natural_embedding_sequence[0, prompt_length:]
        steered_embedding_generated = steered_embedding_sequence[0, prompt_length:]
        
        # Convert to text
        natural_generated_text = self.tokenizer.decode(natural_generated, skip_special_tokens=True)
        steered_generated_text = self.tokenizer.decode(steered_generated, skip_special_tokens=True)
        natural_embedding_generated_text = self.tokenizer.decode(natural_embedding_generated, skip_special_tokens=True)
        steered_embedding_generated_text = self.tokenizer.decode(steered_embedding_generated, skip_special_tokens=True)
        
        return {
            "natural_generated_tokens": natural_generated,
            "steered_generated_tokens": steered_generated,
            "natural_embedding_generated_tokens": natural_embedding_generated,
            "steered_embedding_generated_tokens": steered_embedding_generated,
            "natural_generated_text": natural_generated_text,
            "steered_generated_text": steered_generated_text,
            "natural_embedding_generated_text": natural_embedding_generated_text,
            "steered_embedding_generated_text": steered_embedding_generated_text,
            "generation_steps": generation_steps,
            "full_natural_sequence": natural_sequence[0],
            "full_steered_sequence": steered_sequence[0],
            "full_natural_embedding_sequence": natural_embedding_sequence[0],
            "full_steered_embedding_sequence": steered_embedding_sequence[0]
        }
    
    def evaluate_completion_quality(self, prompt_text: str, token_texts: List[str], 
                                  distribution_type: str) -> List[float]:
        """Use LLM to evaluate how well each token completes the sentence."""
        
        # Format tokens for evaluation
        token_list_str = "\n".join([f"{i+1}. '{token}'" for i, token in enumerate(token_texts)])
        
        # Check if tokens are mostly variations of the same word (for embedding cases)
        diversity_note = ""
        if "embedding" in distribution_type.lower():
            diversity_note = """
IMPORTANT: You are evaluating tokens from an embedding similarity approach, which may produce many variations of the same word (e.g., "History", "history", "Historical", "historians"). When scoring:

- If tokens are mostly variations/inflections of the same word: Base score on completion quality, but apply a diversity penalty (reduce score by 1-2 points) since lack of variety indicates limited expressiveness
- If tokens show good semantic diversity while maintaining quality: Give bonus points (increase score by 1-2 points) for providing meaningful alternatives
- Balance completion quality with lexical diversity - prefer approaches that offer both good completions AND variety in expression"""
        
        evaluation_prompt = f"""You are evaluating how well different tokens complete a sentence fragment.

Sentence fragment: "{prompt_text}"

Please evaluate how well each of the following tokens completes this sentence on a scale of 1-10:
- 1 = Very poor completion (grammatically incorrect, nonsensical, or completely off-topic)
- 5 = Adequate completion (grammatically correct but generic or not particularly fitting)  
- 10 = Excellent completion (grammatically perfect, contextually appropriate, natural, and offers good diversity)

Consider:
- Grammatical correctness
- Contextual appropriateness
- Naturalness and fluency
- Factual accuracy (if applicable)
- Lexical diversity and expressiveness{diversity_note}

Tokens to evaluate:
{token_list_str}

For each token, provide a score and brief reasoning that accounts for both completion quality and diversity."""
        
        
        response = litellm.completion(
            model="gpt-4.1",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.1,
            response_format=CompletionEvaluationResponse
        )
        
        # Parse the structured response
        evaluation_response = CompletionEvaluationResponse.model_validate_json(response.choices[0].message.content)
        
        if evaluation_response and len(evaluation_response.evaluations) <= len(token_texts):
            scores = [eval_item.score for eval_item in evaluation_response.evaluations]
            
            # Validate that tokens match (for debugging)
            for i, (expected_token, eval_item) in enumerate(zip(token_texts, evaluation_response.evaluations)):
                if expected_token.strip() != eval_item.token_text.strip():
                    print(f"⚠️  Token mismatch at position {i}: expected '{expected_token}', got '{eval_item.token_text}'")
            
            return scores
        else:
            print(f"⚠️  Evaluation response is missing tokens: {len(evaluation_response.evaluations)}/{len(token_texts)}")
            raise ValueError(f"Evaluation response is missing tokens: {len(evaluation_response.evaluations)}/{len(token_texts)}")
            
    
    def run_experiment(self, wiki_data: List[Dict], prompt_length: int = 20, 
                      continuation_length: int = 1, top_n: int = 10, output_dir: str = "./results") -> pd.DataFrame:
        """Run the main completion evaluation experiment."""
        
        print(f"🚀 Starting Jacobian completion evaluation experiment")
        print(f"   Wikipedia articles: {len(wiki_data)}")
        print(f"   Prompt length: {prompt_length}")
        print(f"   Continuation length: {continuation_length}")
        print(f"   Top-N tokens: {top_n}")
        
        # Extract completion samples
        completion_samples = self.extract_completion_samples(wiki_data, prompt_length, continuation_length)
        records = []
        detailed_results = []
        
        for i, sample in enumerate(tqdm(completion_samples, desc="Processing samples")):
            sample_id = sample["sample_id"]
            prompt_tokens = sample["prompt_tokens"]
            prompt_text = sample["prompt_text"]
            continuation_length = sample["continuation_length"]
            
            if i % 10 == 0:
                print(f"\n   Processing sample {i+1}/{len(completion_samples)}")
                print(f"   Prompt: '{prompt_text[:50]}...'")
                print(f"   Generating {continuation_length} tokens...")
            
            try:
                # Generate sequences
                target_continuation_tokens = sample["target_continuation_tokens"]
                generation_result = self.generate_sequences(prompt_tokens, target_continuation_tokens)
                natural_generated_text = generation_result["natural_generated_text"]
                steered_generated_text = generation_result["steered_generated_text"]
                natural_embedding_generated_text = generation_result["natural_embedding_generated_text"]
                steered_embedding_generated_text = generation_result["steered_embedding_generated_text"]
                generation_steps = generation_result["generation_steps"]
                
                # Evaluate completion quality using LLM for all four approaches
                natural_scores = self.evaluate_completion_quality(
                    prompt_text, [natural_generated_text], "natural"
                )
                steered_scores = self.evaluate_completion_quality(
                    prompt_text, [steered_generated_text], "steered"
                )
                natural_embedding_scores = self.evaluate_completion_quality(
                    prompt_text, [natural_embedding_generated_text], "natural_embedding"
                )
                steered_embedding_scores = self.evaluate_completion_quality(
                    prompt_text, [steered_embedding_generated_text], "steered_embedding"
                )
                
                # Extract single scores
                natural_score = natural_scores[0] if natural_scores else 5.0
                steered_score = steered_scores[0] if steered_scores else 5.0
                natural_embedding_score = natural_embedding_scores[0] if natural_embedding_scores else 5.0
                steered_embedding_score = steered_embedding_scores[0] if steered_embedding_scores else 5.0
                
                # Calculate metrics from generation steps
                total_steering_success = sum(1 for step in generation_steps if step["steering_result"].get("achieved_top_1", False))
                avg_steering_steps = np.mean([step["steering_result"].get("steps_taken", 0) for step in generation_steps])
                
                # Analyze token-level differences
                token_differences = sum(1 for step in generation_steps if step["natural_token"] != step["steered_token"])
                natural_embed_differences = sum(1 for step in generation_steps if step["natural_token"] != step["natural_embedding_token"])
                steered_embed_differences = sum(1 for step in generation_steps if step["steered_token"] != step["steered_embedding_token"])
                sequence_divergence = token_differences / len(generation_steps) if generation_steps else 0
                
                # Record results
                record = {
                    "sample_id": sample_id,
                    "article_title": sample["article_title"],
                    "prompt_length": len(prompt_tokens),
                    "continuation_length": continuation_length,
                    "natural_generated_text": natural_generated_text,
                    "steered_generated_text": steered_generated_text,
                    "natural_embedding_generated_text": natural_embedding_generated_text,
                    "steered_embedding_generated_text": steered_embedding_generated_text,
                    "natural_score": natural_score,
                    "steered_score": steered_score,
                    "natural_embedding_score": natural_embedding_score,
                    "steered_embedding_score": steered_embedding_score,
                    "score_improvement_steered": steered_score - natural_score,
                    "score_improvement_natural_embedding": natural_embedding_score - natural_score,
                    "score_improvement_steered_embedding": steered_embedding_score - natural_score,
                    "total_steering_success": total_steering_success,
                    "avg_steering_steps": avg_steering_steps,
                    "token_differences": token_differences,
                    "natural_embed_differences": natural_embed_differences,
                    "steered_embed_differences": steered_embed_differences,
                    "sequence_divergence": sequence_divergence,
                    "steering_success_rate": total_steering_success / len(generation_steps) if generation_steps else 0
                }
                
                records.append(record)
                
                # Detailed results for first few samples
                if i < 10:
                    # Convert generation_steps to JSON-serializable format
                    generation_steps_serializable = []
                    for step in generation_steps:
                        step_serializable = {}
                        for key, value in step.items():
                            if isinstance(value, torch.Tensor):
                                if value.numel() == 1:  # Single value tensor
                                    step_serializable[key] = value.item()
                                elif key in ["natural_probs", "steered_probs", "natural_embedding_probs", "steered_embedding_probs"]:
                                    # Store only top-10 tokens for prob distributions
                                    top_probs, top_indices = torch.topk(value, 10)
                                    step_serializable[f"{key}_top10"] = {
                                        "probs": [float(p.item()) for p in top_probs],
                                        "tokens": [int(idx.item()) for idx in top_indices],
                                        "token_texts": [self.tokenizer.decode([idx.item()]) for idx in top_indices]
                                    }
                                else:  # Multi-value tensor - convert to list
                                    step_serializable[key] = value.detach().cpu().tolist()
                            elif isinstance(value, (np.integer, np.int64, np.int32)):
                                step_serializable[key] = int(value)
                            elif isinstance(value, (np.floating, np.float64, np.float32)):
                                step_serializable[key] = float(value)
                            elif isinstance(value, dict):
                                # Recursively convert nested dictionaries
                                step_serializable[key] = self._make_json_serializable(value)
                            else:
                                step_serializable[key] = value
                        generation_steps_serializable.append(step_serializable)
                    
                    detailed_result = {
                        "sample_id": sample_id,
                        "prompt_text": prompt_text,
                        "natural_generated_text": natural_generated_text,
                        "steered_generated_text": steered_generated_text,
                        "natural_embedding_generated_text": natural_embedding_generated_text,
                        "steered_embedding_generated_text": steered_embedding_generated_text,
                        "natural_score": natural_score,
                        "steered_score": steered_score,
                        "natural_embedding_score": natural_embedding_score,
                        "steered_embedding_score": steered_embedding_score,
                        "generation_steps": generation_steps_serializable,
                        "evaluation_method": "sequential_generation_with_llm_evaluation_and_embedding_baselines"
                    }
                    # Ensure entire object is JSON-serializable
                    detailed_result = self._make_json_serializable(detailed_result)
                    detailed_results.append(detailed_result)
                
                if i % 10 == 0:
                    print(f"   Natural score:          {natural_score:.2f}")
                    print(f"   Steered score:          {steered_score:.2f}")
                    print(f"   Natural Embed score:    {natural_embedding_score:.2f}")
                    print(f"   Steered Embed score:    {steered_embedding_score:.2f}")
                    print(f"   Score improvement (steered):        {steered_score - natural_score:.2f}")
                    print(f"   Score improvement (natural embed):  {natural_embedding_score - natural_score:.2f}")
                    print(f"   Score improvement (steered embed):  {steered_embedding_score - natural_score:.2f}")
                    print(f"   Steering success rate: {total_steering_success}/{len(generation_steps)} ({100*total_steering_success/len(generation_steps):.1f}%)")
                    print(f"   Token differences: {token_differences}/{len(generation_steps)} ({100*sequence_divergence:.1f}%)")
                    
            except Exception as e:
                print(f"⚠️  Error processing sample {sample_id}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        csv_path = os.path.join(output_dir, "completion_evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"💾 Results saved to: {csv_path}")
        
        # Save detailed results
        detailed_path = os.path.join(output_dir, "detailed_completion_analysis.json")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        print(f"🔍 Detailed analysis saved to: {detailed_path}")
        
        return df
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Create visualizations for the completion evaluation results."""
        
        print(f"📊 Creating completion evaluation visualizations...")
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comparison plots
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        
        # 1. Score comparison: Natural vs Steered
        ax = axes[0, 0]
        natural_scores = df['natural_score'].values
        steered_scores = df['steered_score'].values
        
        ax.scatter(natural_scores, steered_scores, alpha=0.6, s=50, color='blue')
        ax.plot([0, 10], [0, 10], 'r--', alpha=0.5, label='Equal performance')
        ax.set_xlabel('Natural Score')
        ax.set_ylabel('Steered Score')
        ax.set_title('Natural vs Steered')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Score comparison: Natural vs Natural Embedding
        ax = axes[0, 1]
        natural_embedding_scores = df['natural_embedding_score'].values
        
        ax.scatter(natural_scores, natural_embedding_scores, alpha=0.6, s=50, color='green')
        ax.plot([0, 10], [0, 10], 'r--', alpha=0.5, label='Equal performance')
        ax.set_xlabel('Natural Score')
        ax.set_ylabel('Natural Embedding Score')
        ax.set_title('Natural vs Natural Embedding')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Score comparison: Natural vs Steered Embedding
        ax = axes[0, 2]
        steered_embedding_scores = df['steered_embedding_score'].values
        
        ax.scatter(natural_scores, steered_embedding_scores, alpha=0.6, s=50, color='orange')
        ax.plot([0, 10], [0, 10], 'r--', alpha=0.5, label='Equal performance')
        ax.set_xlabel('Natural Score')
        ax.set_ylabel('Steered Embedding Score')
        ax.set_title('Natural vs Steered Embedding')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Score improvement distributions
        ax = axes[1, 0]
        steered_improvements = df['score_improvement_steered'].values
        natural_embed_improvements = df['score_improvement_natural_embedding'].values
        steered_embed_improvements = df['score_improvement_steered_embedding'].values
        
        ax.hist(steered_improvements, bins=15, alpha=0.6, label='Steered', color='blue')
        ax.hist(natural_embed_improvements, bins=15, alpha=0.6, label='Natural Embed', color='green')
        ax.hist(steered_embed_improvements, bins=15, alpha=0.6, label='Steered Embed', color='orange')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='No improvement')
        ax.set_xlabel('Score Improvement vs Natural')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Score Improvements')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Box plot comparison
        ax = axes[1, 1]
        scores_data = [natural_scores, steered_scores, natural_embedding_scores, steered_embedding_scores]
        labels = ['Natural', 'Steered', 'Natural\nEmbed', 'Steered\nEmbed']
        colors = ['blue', 'red', 'green', 'orange']
        
        bp = ax.boxplot(scores_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Completion Quality Score')
        ax.set_title('Score Distribution Comparison')
        ax.grid(True, alpha=0.3)
        
        # 6. Steering success vs Score improvement
        ax = axes[1, 2]
        ax.scatter(df['steering_success_rate'], df['score_improvement_steered'], alpha=0.6, s=50, color='blue')
        ax.set_xlabel('Steering Success Rate')
        ax.set_ylabel('Score Improvement (Steered)')
        ax.set_title('Steering Success vs Score Improvement')
        ax.grid(True, alpha=0.3)
        
        # 7. Mean score comparison
        ax = axes[2, 0]
        mean_scores = [
            natural_scores.mean(),
            steered_scores.mean(), 
            natural_embedding_scores.mean(),
            steered_embedding_scores.mean()
        ]
        std_scores = [
            natural_scores.std(),
            steered_scores.std(),
            natural_embedding_scores.std(), 
            steered_embedding_scores.std()
        ]
        
        bars = ax.bar(labels, mean_scores, yerr=std_scores, capsize=5, alpha=0.7, color=colors)
        ax.set_ylabel('Mean Completion Score')
        ax.set_title('Mean Scores Comparison')
        ax.grid(True, alpha=0.3)
        
        # 8. Token difference analysis
        ax = axes[2, 1]
        token_diff_data = [
            df['token_differences'],
            df['natural_embed_differences'], 
            df['steered_embed_differences']
        ]
        token_diff_labels = ['Natural vs\nSteered', 'Natural vs\nNat Embed', 'Steered vs\nSteer Embed']
        
        bp = ax.boxplot(token_diff_data, labels=token_diff_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Number of Different Tokens')
        ax.set_title('Token Difference Analysis')
        ax.grid(True, alpha=0.3)
        
        # 9. Correlation heatmap
        ax = axes[2, 2]
        correlation_cols = [
            'natural_score', 'steered_score', 'natural_embedding_score', 'steered_embedding_score',
            'score_improvement_steered', 'score_improvement_natural_embedding', 'score_improvement_steered_embedding',
            'steering_success_rate', 'sequence_divergence'
        ]
        corr_matrix = df[correlation_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Metric Correlations')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "completion_evaluation_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📈 Analysis plot saved to: {plot_path}")
        
        # Create summary statistics plot
        plt.figure(figsize=(14, 8))
        
        # Summary statistics
        plt.subplot(2, 2, 1)
        metrics = ['Natural Score', 'Steered Score']
        values = [
            df['natural_score'].mean(),
            df['steered_score'].mean()
        ]
        errors = [
            df['natural_score'].std(),
            df['steered_score'].std()
        ]
        
        bars = plt.bar(metrics, values, yerr=errors, capsize=5, alpha=0.7)
        plt.title('Overall Performance Comparison')
        plt.ylabel('Completion Quality Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Color code natural vs steered
        bars[0].set_color('blue')
        bars[1].set_color('red')
        
        # Improvement analysis
        plt.subplot(2, 2, 2)
        steered_improvements = df['score_improvement_steered']
        improvement_stats = {
            'Samples with improvement': (steered_improvements > 0).sum(),
            'Samples with degradation': (steered_improvements < 0).sum(),
            'Samples unchanged': (steered_improvements == 0).sum()
        }
        
        plt.pie(improvement_stats.values(), labels=improvement_stats.keys(), 
               autopct='%1.1f%%', startangle=90)
        plt.title('Score Improvement Distribution')
        
        # Sequence divergence analysis
        plt.subplot(2, 2, 3)
        high_divergence = (df['sequence_divergence'] > 0.5).sum()
        low_divergence = (df['sequence_divergence'] <= 0.5).sum()
        
        divergence_stats = {
            'High divergence (>50%)': high_divergence,
            'Low divergence (≤50%)': low_divergence
        }
        
        plt.pie(divergence_stats.values(), labels=divergence_stats.keys(),
               autopct='%1.1f%%', startangle=90)
        plt.title('Sequence Divergence Distribution')
        
        # Steering effectiveness
        plt.subplot(2, 2, 4)
        avg_success_rate = df['steering_success_rate'].mean() * 100
        plt.bar(['Average Steering Success Rate'], [avg_success_rate], alpha=0.7, color='green')
        plt.ylim(0, 100)
        plt.ylabel('Success Rate (%)')
        plt.title(f'Jacobian Steering Success\n({avg_success_rate:.1f}% average success rate)')
        
        plt.tight_layout()
        summary_path = os.path.join(output_dir, "completion_evaluation_summary.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Summary plot saved to: {summary_path}")
        
        print(f"✅ All visualizations created successfully!")
    
    def create_statistics_report(self, df: pd.DataFrame, output_dir: str):
        """Create a comprehensive statistics report with p-values and detailed analysis."""
        
        print(f"📊 Creating statistics report...")
        
        from scipy import stats
        
        # Extract scores
        natural_scores = df['natural_score'].values
        steered_scores = df['steered_score'].values
        natural_embedding_scores = df['natural_embedding_score'].values
        steered_embedding_scores = df['steered_embedding_score'].values
        
        report_path = os.path.join(output_dir, "statistics_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("JACOBIAN COMPLETION EVALUATION - STATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {len(df)}\n\n")
            
            # Descriptive Statistics
            f.write("DESCRIPTIVE STATISTICS\n")
            f.write("-" * 40 + "\n\n")
            
            approaches = [
                ("Natural", natural_scores),
                ("Steered", steered_scores), 
                ("Natural Embedding", natural_embedding_scores),
                ("Steered Embedding", steered_embedding_scores)
            ]
            
            for name, scores in approaches:
                f.write(f"{name}:\n")
                f.write(f"  Mean: {scores.mean():.4f}\n")
                f.write(f"  Std:  {scores.std():.4f}\n")
                f.write(f"  Min:  {scores.min():.4f}\n")
                f.write(f"  Max:  {scores.max():.4f}\n")
                f.write(f"  Median: {np.median(scores):.4f}\n")
                f.write(f"  Q1:   {np.percentile(scores, 25):.4f}\n")
                f.write(f"  Q3:   {np.percentile(scores, 75):.4f}\n\n")
            
            # Pairwise Statistical Tests
            f.write("PAIRWISE STATISTICAL TESTS\n")
            f.write("-" * 40 + "\n\n")
            
            # All pairwise comparisons
            comparisons = [
                ("Natural", "Steered", natural_scores, steered_scores),
                ("Natural", "Natural Embedding", natural_scores, natural_embedding_scores),
                ("Natural", "Steered Embedding", natural_scores, steered_embedding_scores),
                ("Steered", "Natural Embedding", steered_scores, natural_embedding_scores),
                ("Steered", "Steered Embedding", steered_scores, steered_embedding_scores),
                ("Natural Embedding", "Steered Embedding", natural_embedding_scores, steered_embedding_scores)
            ]
            
            for name1, name2, scores1, scores2 in comparisons:
                f.write(f"{name1} vs {name2}:\n")
                
                # Check if we have enough samples for statistical tests
                if len(scores1) < 2:
                    f.write(f"  Insufficient samples (n={len(scores1)}) for statistical tests\n")
                    f.write(f"  Mean difference: {scores2.mean() - scores1.mean():.4f}\n\n")
                    continue
                
                # Paired t-test
                t_stat, t_p_value = stats.ttest_rel(scores2, scores1)
                f.write(f"  Paired t-test:\n")
                f.write(f"    t-statistic: {t_stat:.4f}\n")
                f.write(f"    p-value: {t_p_value:.6f}\n")
                significance = "***" if t_p_value < 0.001 else "**" if t_p_value < 0.01 else "*" if t_p_value < 0.05 else "ns"
                f.write(f"    Significance: {significance}\n")
                f.write(f"    Effect direction: {name2} > {name1}" if t_stat > 0 else f"    Effect direction: {name1} > {name2}\n")
                
                # Wilcoxon signed-rank test (non-parametric) - only if we have enough samples
                try:
                    w_stat, w_p_value = stats.wilcoxon(scores2, scores1, alternative='two-sided')
                    f.write(f"  Wilcoxon signed-rank test:\n")
                    f.write(f"    W-statistic: {w_stat:.4f}\n")
                    f.write(f"    p-value: {w_p_value:.6f}\n")
                    w_significance = "***" if w_p_value < 0.001 else "**" if w_p_value < 0.01 else "*" if w_p_value < 0.05 else "ns"
                    f.write(f"    Significance: {w_significance}\n")
                except ValueError as e:
                    f.write(f"  Wilcoxon test: Not applicable ({e})\n")
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(scores1) - 1) * scores1.var() + (len(scores2) - 1) * scores2.var()) / (len(scores1) + len(scores2) - 2))
                cohens_d = (scores2.mean() - scores1.mean()) / pooled_std
                f.write(f"  Effect size (Cohen's d): {cohens_d:.4f}\n")
                
                effect_magnitude = "negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
                f.write(f"  Effect magnitude: {effect_magnitude}\n\n")
            
            # Score Improvements Analysis
            f.write("SCORE IMPROVEMENTS ANALYSIS\n")
            f.write("-" * 40 + "\n\n")
            
            improvements = [
                ("Steered vs Natural", df['score_improvement_steered'].values),
                ("Natural Embedding vs Natural", df['score_improvement_natural_embedding'].values),
                ("Steered Embedding vs Natural", df['score_improvement_steered_embedding'].values)
            ]
            
            for name, improvement_scores in improvements:
                f.write(f"{name}:\n")
                f.write(f"  Mean improvement: {improvement_scores.mean():.4f}\n")
                f.write(f"  Std improvement:  {improvement_scores.std():.4f}\n")
                f.write(f"  Samples with improvement (>0): {(improvement_scores > 0).sum()}/{len(improvement_scores)} ({(improvement_scores > 0).mean()*100:.1f}%)\n")
                f.write(f"  Samples with degradation (<0): {(improvement_scores < 0).sum()}/{len(improvement_scores)} ({(improvement_scores < 0).mean()*100:.1f}%)\n")
                f.write(f"  Samples unchanged (=0): {(improvement_scores == 0).sum()}/{len(improvement_scores)} ({(improvement_scores == 0).mean()*100:.1f}%)\n")
                
                # One-sample t-test against 0 (no improvement)
                t_stat_improvement, t_p_improvement = stats.ttest_1samp(improvement_scores, 0)
                f.write(f"  One-sample t-test (H0: improvement = 0):\n")
                f.write(f"    t-statistic: {t_stat_improvement:.4f}\n")
                f.write(f"    p-value: {t_p_improvement:.6f}\n")
                improvement_significance = "***" if t_p_improvement < 0.001 else "**" if t_p_improvement < 0.01 else "*" if t_p_improvement < 0.05 else "ns"
                f.write(f"    Significance: {improvement_significance}\n\n")
            
            # Correlation Analysis
            f.write("CORRELATION ANALYSIS\n")
            f.write("-" * 40 + "\n\n")
            
            # Key correlations
            key_correlations = [
                ("Steering Success Rate vs Score Improvement (Steered)", 
                 df['steering_success_rate'], df['score_improvement_steered']),
                ("Natural Score vs Natural Embedding Score", 
                 df['natural_score'], df['natural_embedding_score']),
                ("Steered Score vs Steered Embedding Score", 
                 df['steered_score'], df['steered_embedding_score']),
                ("Sequence Divergence vs Score Improvement (Steered)", 
                 df['sequence_divergence'], df['score_improvement_steered'])
            ]
            
            for name, var1, var2 in key_correlations:
                f.write(f"{name}:\n")
                if len(var1) < 3:  # Need at least 3 points for meaningful correlation
                    f.write(f"  Insufficient samples (n={len(var1)}) for correlation analysis\n\n")
                    continue
                    
                try:
                    corr_coeff, corr_p_value = stats.pearsonr(var1, var2)
                    f.write(f"  Pearson correlation: {corr_coeff:.4f}\n")
                    f.write(f"  p-value: {corr_p_value:.6f}\n")
                    corr_significance = "***" if corr_p_value < 0.001 else "**" if corr_p_value < 0.01 else "*" if corr_p_value < 0.05 else "ns"
                    f.write(f"  Significance: {corr_significance}\n\n")
                except Exception as e:
                    f.write(f"  Correlation analysis failed: {e}\n\n")
            
            # Summary Conclusions
            f.write("SUMMARY CONCLUSIONS\n")
            f.write("-" * 40 + "\n\n")
            
            # Best performing approach
            mean_scores = [natural_scores.mean(), steered_scores.mean(), 
                          natural_embedding_scores.mean(), steered_embedding_scores.mean()]
            approach_names = ["Natural", "Steered", "Natural Embedding", "Steered Embedding"]
            best_idx = np.argmax(mean_scores)
            f.write(f"Best performing approach: {approach_names[best_idx]} (mean score: {mean_scores[best_idx]:.4f})\n")
            
            # Significant improvements
            significant_improvements = []
            for name, improvement_scores in improvements:
                t_stat, p_val = stats.ttest_1samp(improvement_scores, 0)
                if p_val < 0.05 and t_stat > 0:
                    significant_improvements.append(name)
            
            if significant_improvements:
                f.write(f"Approaches with significant improvements: {', '.join(significant_improvements)}\n")
            else:
                f.write("No approaches showed statistically significant improvements.\n")
            
            # Embedding vs LM Head comparison
            natural_vs_embed = stats.ttest_rel(natural_embedding_scores, natural_scores)
            steered_vs_embed = stats.ttest_rel(steered_embedding_scores, steered_scores)
            
            f.write(f"\nEmbedding vs LM Head Analysis:\n")
            f.write(f"  Natural Embedding vs Natural LM Head: p={natural_vs_embed[1]:.6f}\n")
            f.write(f"  Steered Embedding vs Steered LM Head: p={steered_vs_embed[1]:.6f}\n")
            
            f.write(f"\nNote: Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05\n")
        
        print(f"   📊 Statistics report saved to: {report_path}")
        return report_path
    
    def print_summary(self, df: pd.DataFrame):
        """Print a summary of the experiment results."""
        
        print("\n" + "="*80)
        print("📋 JACOBIAN COMPLETION EVALUATION SUMMARY")
        print("="*80)
        
        print(f"Total samples processed: {len(df)}")
        
        # Overall performance metrics
        print(f"\n📊 COMPLETION QUALITY SCORES:")
        print(f"   Natural score:          {df['natural_score'].mean():.3f} ± {df['natural_score'].std():.3f}")
        print(f"   Steered score:          {df['steered_score'].mean():.3f} ± {df['steered_score'].std():.3f}")
        print(f"   Natural Embed score:    {df['natural_embedding_score'].mean():.3f} ± {df['natural_embedding_score'].std():.3f}")
        print(f"   Steered Embed score:    {df['steered_embedding_score'].mean():.3f} ± {df['steered_embedding_score'].std():.3f}")
        
        # Improvement analysis
        steered_improvements = df['score_improvement_steered']
        natural_embed_improvements = df['score_improvement_natural_embedding']
        steered_embed_improvements = df['score_improvement_steered_embedding']
        
        print(f"\n🎯 IMPROVEMENT ANALYSIS:")
        print(f"   Steered vs Natural:")
        print(f"     Average improvement: {steered_improvements.mean():.3f} ± {steered_improvements.std():.3f}")
        print(f"     Samples improved: {(steered_improvements > 0).sum()}/{len(df)} ({(steered_improvements > 0).mean()*100:.1f}%)")
        
        print(f"   Natural Embedding vs Natural:")
        print(f"     Average improvement: {natural_embed_improvements.mean():.3f} ± {natural_embed_improvements.std():.3f}")
        print(f"     Samples improved: {(natural_embed_improvements > 0).sum()}/{len(df)} ({(natural_embed_improvements > 0).mean()*100:.1f}%)")
        
        print(f"   Steered Embedding vs Natural:")
        print(f"     Average improvement: {steered_embed_improvements.mean():.3f} ± {steered_embed_improvements.std():.3f}")
        print(f"     Samples improved: {(steered_embed_improvements > 0).sum()}/{len(df)} ({(steered_embed_improvements > 0).mean()*100:.1f}%)")
        
        # Sequential generation analysis
        print(f"\n🔄 SEQUENTIAL GENERATION ANALYSIS:")
        print(f"   Average continuation length: {df['continuation_length'].mean():.1f}")
        print(f"   Average steering success rate: {df['steering_success_rate'].mean()*100:.1f}%")
        print(f"   Average steering steps per token: {df['avg_steering_steps'].mean():.1f}")
        print(f"   Average token differences: {df['token_differences'].mean():.1f}")
        print(f"   Average sequence divergence: {df['sequence_divergence'].mean()*100:.1f}%")
        
        # Analyze relationship between steering success and score improvement
        high_steering_success = df[df['steering_success_rate'] > 0.5]
        low_steering_success = df[df['steering_success_rate'] <= 0.5]
        
        if len(high_steering_success) > 0 and len(low_steering_success) > 0:
            print(f"\n⚙️ STEERING EFFECTIVENESS:")
            print(f"   Samples with high steering success (>50%): {len(high_steering_success)}")
            print(f"   Samples with low steering success (≤50%): {len(low_steering_success)}")
            print(f"   Score improvement (high steering success): {high_steering_success['score_improvement_steered'].mean():.3f}")
            print(f"   Score improvement (low steering success): {low_steering_success['score_improvement_steered'].mean():.3f}")
        
        # Divergence analysis
        print(f"\n🌟 SEQUENCE DIVERGENCE:")
        high_divergence = df[df['sequence_divergence'] > 0.5]
        low_divergence = df[df['sequence_divergence'] <= 0.5]
        
        if len(high_divergence) > 0 and len(low_divergence) > 0:
            print(f"   High divergence samples (>50% different tokens): {len(high_divergence)}")
            print(f"   Low divergence samples (≤50% different tokens): {len(low_divergence)}")
            print(f"   Score improvement (high divergence): {high_divergence['score_improvement_steered'].mean():.3f}")
            print(f"   Score improvement (low divergence): {low_divergence['score_improvement_steered'].mean():.3f}")
        
        # Statistical significance
        from scipy import stats
        
        print(f"\n📊 STATISTICAL TESTS:")
        
        # Extract all scores
        natural_scores = df['natural_score'].values
        steered_scores = df['steered_score'].values
        natural_embedding_scores = df['natural_embedding_score'].values
        steered_embedding_scores = df['steered_embedding_score'].values
        
        # Test each approach vs natural
        comparisons = [
            ("Steered vs Natural", steered_scores, natural_scores),
            ("Natural Embedding vs Natural", natural_embedding_scores, natural_scores),
            ("Steered Embedding vs Natural", steered_embedding_scores, natural_scores)
        ]
        
        for name, scores1, scores2 in comparisons:
            print(f"   {name}:")
            
            # Check if we have enough samples for statistical tests
            if len(scores1) < 2:
                print(f"     Insufficient samples (n={len(scores1)}) for statistical tests")
                print(f"     Mean difference: {scores1.mean() - scores2.mean():.3f}")
                continue
            
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            print(f"     Paired t-test: t={t_stat:.3f}, p={p_value:.6f}")
            
            try:
                w_stat, w_p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')
                print(f"     Wilcoxon test: W={w_stat:.3f}, p={w_p_value:.6f}")
            except ValueError:
                print(f"     Wilcoxon test: Not applicable (insufficient data)")
            
            significance = "✅ Yes" if p_value < 0.05 else "❌ No"
            direction = "improvement" if t_stat > 0 else "degradation"
            print(f"     Significant {direction}: {significance}")
        
        # Correlation analysis
        print(f"\n🔗 CORRELATIONS:")
        steering_steered_corr = df['steering_success_rate'].corr(df['score_improvement_steered'])
        divergence_steered_corr = df['sequence_divergence'].corr(df['score_improvement_steered'])
        natural_embedding_corr = df['natural_score'].corr(df['natural_embedding_score'])
        steered_embedding_corr = df['steered_score'].corr(df['steered_embedding_score'])
        
        print(f"   Steering success vs Steered improvement: {steering_steered_corr:.3f}")
        print(f"   Sequence divergence vs Steered improvement: {divergence_steered_corr:.3f}")
        print(f"   Natural vs Natural Embedding scores: {natural_embedding_corr:.3f}")
        print(f"   Steered vs Steered Embedding scores: {steered_embedding_corr:.3f}")


def main():
    """Main function to run the Jacobian completion evaluation experiment."""
    
    parser = argparse.ArgumentParser(description="🚀 Jacobian Steering Completion Evaluation")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="HuggingFace model ID")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of Wikipedia articles to sample")
    parser.add_argument("--prompt_length", type=int, default=20,
                       help="Length of prompt in tokens")
    parser.add_argument("--continuation_length", type=int, default=5,
                       help="Length of continuation in tokens to evaluate")
    parser.add_argument("--top_n", type=int, default=10,
                       help="Number of top tokens to evaluate from each distribution")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu", "mps"],
                       help="Device to use: cuda, cpu, or mps")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Where to write results")
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"jacobian_completion_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Starting Jacobian Completion Evaluation Experiment")
    print(f"   Model: {args.model_name}")
    print(f"   Wikipedia samples: {args.num_samples}")
    print(f"   Prompt length: {args.prompt_length}")
    print(f"   Continuation length: {args.continuation_length}")
    print(f"   Top-N evaluation: {args.top_n}")
    print(f"   Device: {args.device}")
    print(f"   Output directory: {output_dir}")
    
    # Initialize experiment
    experiment = JacobianCompletionExperiment(
        model_name=args.model_name, 
        device=args.device
    )
    
    # Load Wikipedia data
    wiki_data = experiment.load_wikipedia_data(num_samples=args.num_samples)
    
    # Run experiment
    df = experiment.run_experiment(
        wiki_data=wiki_data,
        prompt_length=args.prompt_length,
        continuation_length=args.continuation_length,
        top_n=args.top_n,
        output_dir=output_dir
    )
    
    # Create visualizations
    experiment.create_visualizations(df, output_dir)
    
    # Create statistics report
    experiment.create_statistics_report(df, output_dir)
    
    # Print summary
    experiment.print_summary(df)
    
    # Save experiment configuration
    config = {
        "model_name": args.model_name,
        "num_samples": args.num_samples,
        "prompt_length": args.prompt_length,
        "continuation_length": args.continuation_length,
        "top_n": args.top_n,
        "device": args.device,
        "timestamp": timestamp,
        "total_completion_samples": len(df),
        "output_dir": output_dir,
        "output_files": {
            "results": "completion_evaluation_results.csv",
            "detailed_analysis": "detailed_completion_analysis.json",
            "config": "experiment_config.json"
        }
    }
    
    config_path = os.path.join(output_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Jacobian completion evaluation experiment completed successfully!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 