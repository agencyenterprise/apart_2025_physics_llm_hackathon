#!/usr/bin/env python3
"""
Momentum-Perplexity Phase Space Analysis Experiment
==================================================

This experiment treats language model generation as a dynamical system where:
- Momentum: Rate of change in hidden space (like velocity)
- Perplexity: Uncertainty/confidence (like potential energy)

We analyze trajectories in this 2D phase space to find:
- Attractors (regions trajectories tend toward)
- Cycles (repeated patterns)  
- Flow patterns (how trajectories move through phase space)
- Fixed points (equilibrium states)
- Strange attractors (chaotic patterns)

Phase Space Coordinates:
- X-axis: Momentum (||h_t - h_{t-1}||)
- Y-axis: Perplexity (point or distribution)

Usage Examples:
===============
# Basic phase space analysis
python momentum_perplexity_phase_space_experiment.py --num_articles 20

# Use random-weight baseline for comparison
python momentum_perplexity_phase_space_experiment.py --model_name "Qwen/Qwen2.5-1.5B-Instruct:random" --num_articles 20

# Focus on specific perplexity range
python momentum_perplexity_phase_space_experiment.py --perplexity_cutoff 50 --num_articles 30

# Detailed analysis with longer trajectories  
python momentum_perplexity_phase_space_experiment.py --max_steps_per_text 15 --num_articles 25

# Compare pretrained vs random baseline
python momentum_perplexity_phase_space_experiment.py --model_name "Qwen/Qwen2.5-1.5B-Instruct" --num_articles 10
python momentum_perplexity_phase_space_experiment.py --model_name "Qwen/Qwen2.5-1.5B-Instruct:random" --num_articles 10

# Use generation mode to create trajectories through token generation
python momentum_perplexity_phase_space_experiment.py --generate_tokens 15 --num_articles 10

# Generation mode with random baseline
python momentum_perplexity_phase_space_experiment.py --model_name "Qwen/Qwen2.5-1.5B-Instruct:random" --generate_tokens 20 --num_articles 5
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
# Visualization imports moved to phase_space_visualizer.py
import argparse
import datasets
from pathlib import Path
# Scipy and matplotlib imports moved to phase_space_visualizer.py
import string
import re
import sys
import os
import platform
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.experiment_4_momentum_perplexity_band.phase_space_visualizer import PhaseSpaceVisualizer
from utils.model_utils import load_random_weight_model

class MomentumPerplexityPhaseSpaceExperiment:
    """Analyze language model trajectories in momentum-perplexity phase space"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", 
                 device: str = "cpu",
                 generate_tokens: Optional[int] = None):
        
        self.model_name = model_name
        self.device = device
        self.generate_tokens = generate_tokens
        
        # Check if we should use random weights
        if model_name.endswith(":random"):
            base_model_name = model_name[:-7]  # Remove ":random" suffix
            print(f"🎲 Loading random-weight baseline for: {base_model_name}")
            self.model, self.tokenizer, self.config = load_random_weight_model(
                model_name=base_model_name,
                device=device,
                torch_dtype=torch.float32
            )
            self.base_model_name = base_model_name
            self.is_random_model = True
        else:
            print(f"🤖 Loading pretrained model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.base_model_name = model_name
            self.is_random_model = False
        
        print(f"✅ Model loaded, Hidden dim: {self.model.config.hidden_size}")
        if self.is_random_model:
            print(f"🔄 Using random-weight baseline (architecture from {self.base_model_name})")
        if self.generate_tokens:
            print(f"🎯 Generation mode: Will generate {self.generate_tokens} additional tokens per trajectory")
        print(f"🌌 Phase Space Analysis: Momentum vs Perplexity")
    
    def load_wikipedia_samples(self, num_samples: int = 50) -> List[str]:
        """Load Wikipedia articles using the datasets library"""
        print(f"📚 Loading {num_samples} Wikipedia articles from datasets...")
        
        try:
            dataset = datasets.load_dataset(
                "wikipedia", 
                "20220301.en", 
                split=f"train[:{num_samples*3}]",
                trust_remote_code=True
            )
            
            articles = []
            for i, item in enumerate(dataset):
                if len(articles) >= num_samples:
                    break
                    
                text = item.get('text', '')
                title = item.get('title', '')
                
                if text and len(text) > 200:
                    sentences = text.split('. ')[:4]  # First 4 sentences
                    clean_text = '. '.join(sentences).strip()
                    
                    if len(clean_text) > 100:
                        articles.append(clean_text)
                        print(f"  {len(articles)}/{num_samples}: {title[:50]}...")
            
            print(f"✅ Loaded {len(articles)} articles from Wikipedia dataset")
            return articles
            
        except Exception as e:
            print(f"❌ Failed to load Wikipedia dataset: {e}")
            raise RuntimeError(f"Could not load Wikipedia dataset: {e}")
    
    def extract_hidden_states(self, token_ids: List[int]) -> torch.Tensor:
        """Extract hidden states for a sequence of tokens"""
        with torch.no_grad():
            input_ids = torch.tensor([token_ids]).to(self.device)
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][0]
        return hidden_states.cpu()
    
    def generate_tokens_manually(self, initial_tokens: List[int], num_generate: int, temperature: float = 1.0) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Generate tokens manually using forward passes and collect hidden states.
        
        Returns:
            generated_tokens: List of all tokens (initial + generated)
            all_hidden_states: Hidden states for each token position
        """
        print(f"    🎯 Generating {num_generate} tokens manually with temperature={temperature}")
        
        current_tokens = initial_tokens.copy()
        all_hidden_states = []
        
        with torch.no_grad():
            # First, get hidden states for initial tokens
            input_ids = torch.tensor([current_tokens]).to(self.device)
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
            
            # Store all initial hidden states
            for i in range(len(current_tokens)):
                all_hidden_states.append(hidden_states[i].cpu())
            
            # Now generate new tokens one by one
            for gen_step in range(num_generate):
                # Get logits for next token prediction (from last position)
                logits = outputs.logits[0, -1, :]  # Last position logits
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Get top-1 token (greedy decoding)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.argmax(logits).item()
                
                # Add to sequence
                current_tokens.append(next_token)
                
                # Debug info
                next_token_text = self.tokenizer.decode([next_token])
                print(f"      Step {gen_step + 1}/{num_generate}: '{next_token_text}' (prob: {probs[next_token]:.3f})")
                
                # Get hidden state for the newly generated token
                input_ids = torch.tensor([current_tokens]).to(self.device)
                outputs = self.model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
                
                # Store hidden state for the new token (last position)
                all_hidden_states.append(hidden_states[-1].cpu())
        
        return current_tokens, all_hidden_states
    
    def calculate_perplexities(self, hidden_state: np.ndarray) -> Tuple[float, float]:
        """Calculate both perplexity metrics"""
        with torch.no_grad():
            h_tensor = torch.tensor(hidden_state, dtype=torch.float32).to(self.device)
            logits = self.model.lm_head(h_tensor)
            probs = F.softmax(logits, dim=-1)
            
            # Point perplexity: 1/p_highest_prob_token
            max_prob = torch.max(probs).item()
            point_perplexity = 1.0 / max_prob
            
            # Distribution perplexity: exp(-sum(px*log(px)))
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            distribution_perplexity = torch.exp(entropy).item()
            
        return point_perplexity, distribution_perplexity
    
    def analyze_token_type(self, token_text: str) -> str:
        """Classify token into categories for analysis"""
        token_clean = token_text.strip()
        
        if not token_clean:
            return 'empty'
        
        # Punctuation
        if all(c in string.punctuation for c in token_clean):
            return 'punctuation'
        
        # Numbers
        if re.match(r'^[\d\.,]+$', token_clean):
            return 'number'
        
        # Function words (common short words)
        function_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'down', 'out', 'off', 'over', 'under',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those'
        }
        
        if token_clean.lower() in function_words:
            return 'function_word'
        
        # Capitalized words (proper nouns, sentence starts)
        if token_clean[0].isupper() and len(token_clean) > 1:
            return 'capitalized'
        
        # Long words (likely content words)
        if len(token_clean) >= 6:
            return 'long_content'
        
        # Short content words
        if len(token_clean) >= 2:
            return 'short_content'
        
        return 'other'
    
    def calculate_energy_measures(self, momentum: float, point_perp: float, dist_perp: float) -> Dict[str, float]:
        """Calculate energy-like measures for phase space dynamics"""
        
        # Kinetic-like energy from momentum
        kinetic_energy = (momentum ** 2) / 2.0
        
        # Potential-like energy from perplexity (uncertainty)
        point_potential = np.log(max(point_perp, 1e-8))
        dist_potential = np.log(max(dist_perp, 1e-8))
        
        # Total energy measures
        total_energy_point = kinetic_energy + point_potential
        total_energy_dist = kinetic_energy + dist_potential
        
        # Normalized energies (0-1 scale for visualization)
        energy_ratio = kinetic_energy / (kinetic_energy + point_potential + 1e-8)
        
        return {
            'kinetic_energy': kinetic_energy,
            'point_potential': point_potential,
            'dist_potential': dist_potential,
            'total_energy_point': total_energy_point,
            'total_energy_dist': total_energy_dist,
            'energy_ratio': energy_ratio
        }
    
    def calculate_velocity_acceleration(self, trajectory_momentum: List[float], trajectory_point_perp: List[float], trajectory_dist_perp: List[float]) -> Tuple[List[Dict], List[Dict]]:
        """Calculate velocity and acceleration vectors in phase space"""
        
        velocities = []
        accelerations = []
        
        for i in range(len(trajectory_momentum)):
            if i == 0:
                # First step: no velocity/acceleration
                velocities.append({
                    'momentum_velocity': 0.0,
                    'point_perp_velocity': 0.0,
                    'dist_perp_velocity': 0.0,
                    'velocity_magnitude': 0.0
                })
                accelerations.append({
                    'momentum_acceleration': 0.0,
                    'point_perp_acceleration': 0.0,
                    'dist_perp_acceleration': 0.0,
                    'acceleration_magnitude': 0.0
                })
            else:
                # Calculate velocity (first derivative)
                momentum_vel = trajectory_momentum[i] - trajectory_momentum[i-1]
                point_perp_vel = trajectory_point_perp[i] - trajectory_point_perp[i-1]
                dist_perp_vel = trajectory_dist_perp[i] - trajectory_dist_perp[i-1]
                
                vel_magnitude = np.sqrt(momentum_vel**2 + point_perp_vel**2 + dist_perp_vel**2)
                
                velocities.append({
                    'momentum_velocity': momentum_vel,
                    'point_perp_velocity': point_perp_vel,
                    'dist_perp_velocity': dist_perp_vel,
                    'velocity_magnitude': vel_magnitude
                })
                
                if i == 1:
                    # Second step: no acceleration yet
                    accelerations.append({
                        'momentum_acceleration': 0.0,
                        'point_perp_acceleration': 0.0,
                        'dist_perp_acceleration': 0.0,
                        'acceleration_magnitude': 0.0
                    })
                else:
                    # Calculate acceleration (second derivative)
                    prev_momentum_vel = trajectory_momentum[i-1] - trajectory_momentum[i-2]
                    prev_point_perp_vel = trajectory_point_perp[i-1] - trajectory_point_perp[i-2]
                    prev_dist_perp_vel = trajectory_dist_perp[i-1] - trajectory_dist_perp[i-2]
                    
                    momentum_acc = momentum_vel - prev_momentum_vel
                    point_perp_acc = point_perp_vel - prev_point_perp_vel
                    dist_perp_acc = dist_perp_vel - prev_dist_perp_vel
                    
                    acc_magnitude = np.sqrt(momentum_acc**2 + point_perp_acc**2 + dist_perp_acc**2)
                    
                    accelerations.append({
                        'momentum_acceleration': momentum_acc,
                        'point_perp_acceleration': point_perp_acc,
                        'dist_perp_acceleration': dist_perp_acc,
                        'acceleration_magnitude': acc_magnitude
                    })
        
        return velocities, accelerations 
    
    def extract_trajectory_phase_space(self, text: str, max_steps: int = 10) -> Optional[Dict]:
        """Extract a single trajectory in momentum-perplexity phase space"""
        
        print(f"🌌 Extracting phase space trajectory: '{text[:60]}...'")
        
        # Tokenize text
        initial_tokens = self.tokenizer.encode(text[:300], add_special_tokens=False)
        if len(initial_tokens) < 3:
            print("  ⚠️ Text too short, skipping")
            return None
        
        try:
            if self.generate_tokens and self.generate_tokens > 0:
                # Generation mode: use initial tokens as prompt and generate additional tokens
                prompt_length = min(len(initial_tokens), max(3, max_steps // 2))  # Use part of text as prompt
                prompt_tokens = initial_tokens[:prompt_length]
                
                print(f"    🎯 Using {prompt_length} tokens as prompt, generating {self.generate_tokens} tokens")
                
                # Generate tokens manually and get all hidden states
                all_tokens, hidden_states = self.generate_tokens_manually(
                    prompt_tokens, 
                    self.generate_tokens, 
                    temperature=1
                )
                trajectory_tokens = all_tokens
                
                print(f"    ✅ Generated trajectory with {len(hidden_states)} hidden states")
            else:
                # Standard mode: use existing tokens up to max_steps
                max_trajectory_length = min(len(initial_tokens), max_steps + 2)
                trajectory_tokens = initial_tokens[:max_trajectory_length]
                prompt_tokens = trajectory_tokens  # In standard mode, all tokens are "prompt"
                
                # Extract hidden states for entire trajectory
                hidden_states = self.extract_hidden_states(trajectory_tokens)
                print(f"    🎯 Extracted {len(hidden_states)} hidden states")
            
            # Calculate phase space coordinates for each step
            trajectory = {
                'text': text[:300],
                'token_sequence': trajectory_tokens,
                'generation_used': self.generate_tokens is not None and self.generate_tokens > 0,
                'num_generated_tokens': self.generate_tokens if self.generate_tokens else 0,
                'prompt_length': len(prompt_tokens) if self.generate_tokens and self.generate_tokens > 0 else len(trajectory_tokens),
                'steps': [],
                'momentum': [],
                'point_perplexity': [],
                'distribution_perplexity': [],
                'hidden_states': [],
                # Tier 1 additions
                'token_types': [],
                'energy_measures': [],
                'velocities': [],
                'accelerations': []
            }
            
            for step in range(len(hidden_states)):
                current_hidden = hidden_states[step].numpy()
                
                # Calculate momentum (0 for first step)
                if step == 0:
                    momentum = 0.0
                else:
                    prev_hidden = hidden_states[step-1].numpy()
                    momentum = np.linalg.norm(current_hidden - prev_hidden)
                
                # Calculate perplexities
                point_perp, dist_perp = self.calculate_perplexities(current_hidden)
                
                # Tier 1 Analysis: Token type
                token_text = self.tokenizer.decode([trajectory_tokens[step]]) if step < len(trajectory_tokens) else ''
                token_type = self.analyze_token_type(token_text)
                
                # Determine if this token was generated or from original prompt
                is_generated = False
                if self.generate_tokens and self.generate_tokens > 0:
                    prompt_length = len(prompt_tokens)
                    is_generated = step >= prompt_length
                
                # Tier 1 Analysis: Energy measures
                energy_measures = self.calculate_energy_measures(momentum, point_perp, dist_perp)
                
                # Store phase space point with Tier 1 data
                step_data = {
                    'step': step,
                    'momentum': momentum,
                    'point_perplexity': point_perp,
                    'distribution_perplexity': dist_perp,
                    'token': token_text,
                    'token_type': token_type,
                    'is_generated': is_generated,
                    'energy_measures': energy_measures,
                    'hidden_state': current_hidden
                }
                
                trajectory['steps'].append(step_data)
                trajectory['momentum'].append(momentum)
                trajectory['point_perplexity'].append(point_perp)
                trajectory['distribution_perplexity'].append(dist_perp)
                trajectory['hidden_states'].append(current_hidden)
                trajectory['token_types'].append(token_type)
                trajectory['energy_measures'].append(energy_measures)
                
                generated_flag = "🎯" if is_generated else "📝"
                print(f"    Step {step}: {generated_flag} momentum={momentum:.1f}, point_perp={point_perp:.2f}, dist_perp={dist_perp:.2f}, token='{token_text}', type={token_type}")
            
            # Tier 1 Analysis: Calculate velocity and acceleration after all steps
            velocities, accelerations = self.calculate_velocity_acceleration(
                trajectory['momentum'], 
                trajectory['point_perplexity'], 
                trajectory['distribution_perplexity']
            )
            
            trajectory['velocities'] = velocities
            trajectory['accelerations'] = accelerations
            
            # Add velocity/acceleration to step data
            for i, step_data in enumerate(trajectory['steps']):
                if i < len(velocities):
                    step_data['velocity'] = velocities[i]
                if i < len(accelerations):
                    step_data['acceleration'] = accelerations[i]
            
            return trajectory
            
        except Exception as e:
            print(f"    ❌ Error extracting trajectory: {e}")
            return None
    
    def extract_all_trajectories(self, text_samples: List[str], max_steps_per_text: int = 10) -> List[Dict]:
        """Extract all trajectories for phase space analysis"""
        
        print("🌌 PHASE SPACE TRAJECTORY EXTRACTION")
        print("=" * 60)
        
        all_trajectories = []
        
        for i, text in enumerate(text_samples):
            print(f"\n📝 TEXT {i+1}/{len(text_samples)}")
            
            try:
                trajectory = self.extract_trajectory_phase_space(text, max_steps_per_text)
                if trajectory is not None:
                    all_trajectories.append(trajectory)
                    print(f"  ✅ Trajectory: {len(trajectory['steps'])} steps")
                else:
                    print(f"  ❌ Failed to extract trajectory")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        print(f"\n📊 EXTRACTION COMPLETE:")
        print(f"🎯 Total trajectories: {len(all_trajectories)}")
        
        # Calculate aggregate statistics
        all_momentum = []
        all_point_perp = []
        all_dist_perp = []
        
        for traj in all_trajectories:
            all_momentum.extend(traj['momentum'][1:])  # Skip first step (momentum=0)
            all_point_perp.extend(traj['point_perplexity'])
            all_dist_perp.extend(traj['distribution_perplexity'])
        
        print(f"📈 Phase space coverage:")
        print(f"   Momentum range: {np.min(all_momentum):.1f} - {np.max(all_momentum):.1f}")
        print(f"   Point perplexity range: {np.min(all_point_perp):.2f} - {np.max(all_point_perp):.2f}")
        print(f"   Distribution perplexity range: {np.min(all_dist_perp):.2f} - {np.max(all_dist_perp):.2f}")
        
        return all_trajectories 
    
    def create_phase_space_visualizations(self, trajectories: List[Dict], output_dir: str, perplexity_cutoff: Optional[float] = None) -> None:
        """Create comprehensive phase space visualizations using the visualization library"""
        
        print(f"🎨 Creating phase space visualizations...")
        
        # Initialize visualizer and create all visualizations
        visualizer = PhaseSpaceVisualizer()
        visualizer.create_all_visualizations(trajectories, output_dir, perplexity_cutoff)
        
        print(f"  ✅ Phase space visualizations completed using visualization library")
    
    def analyze_phase_space_statistics(self, trajectories: List[Dict]) -> Dict:
        """Analyze statistical properties of the phase space"""
        
        print("📊 Analyzing phase space statistics...")
        
        # Collect all phase space points
        all_momentum = []
        all_point_perp = []
        all_dist_perp = []
        
        trajectory_stats = []
        
        for i, traj in enumerate(trajectories):
            momentum = np.array(traj['momentum'][1:])  # Skip first step
            point_perp = np.array(traj['point_perplexity'][1:])
            dist_perp = np.array(traj['distribution_perplexity'][1:])
            
            all_momentum.extend(momentum)
            all_point_perp.extend(point_perp)
            all_dist_perp.extend(dist_perp)
            
            # Per-trajectory statistics
            if len(momentum) > 0:
                traj_stats = {
                    'trajectory_id': i,
                    'length': len(momentum),
                    'momentum_range': [float(np.min(momentum)), float(np.max(momentum))],
                    'momentum_mean': float(np.mean(momentum)),
                    'momentum_std': float(np.std(momentum)),
                    'point_perp_range': [float(np.min(point_perp)), float(np.max(point_perp))],
                    'point_perp_mean': float(np.mean(point_perp)),
                    'dist_perp_range': [float(np.min(dist_perp)), float(np.max(dist_perp))],
                    'dist_perp_mean': float(np.mean(dist_perp)),
                    'momentum_vs_point_perp_correlation': float(np.corrcoef(momentum, point_perp)[0,1]) if len(momentum) > 1 else 0.0,
                    'momentum_vs_dist_perp_correlation': float(np.corrcoef(momentum, dist_perp)[0,1]) if len(momentum) > 1 else 0.0
                }
                trajectory_stats.append(traj_stats)
        
        # Global statistics
        all_momentum = np.array(all_momentum)
        all_point_perp = np.array(all_point_perp)
        all_dist_perp = np.array(all_dist_perp)
        
        global_stats = {
            'total_trajectories': len(trajectories),
            'total_phase_space_points': len(all_momentum),
            'phase_space_coverage': {
                'momentum_range': [float(np.min(all_momentum)), float(np.max(all_momentum))],
                'point_perplexity_range': [float(np.min(all_point_perp)), float(np.max(all_point_perp))],
                'distribution_perplexity_range': [float(np.min(all_dist_perp)), float(np.max(all_dist_perp))]
            },
            'global_correlations': {
                'momentum_vs_point_perplexity': float(np.corrcoef(all_momentum, all_point_perp)[0,1]),
                'momentum_vs_distribution_perplexity': float(np.corrcoef(all_momentum, all_dist_perp)[0,1]),
                'point_vs_distribution_perplexity': float(np.corrcoef(all_point_perp, all_dist_perp)[0,1])
            },
            'phase_space_density': {
                'momentum_density_peaks': self._find_density_peaks(all_momentum),
                'point_perplexity_density_peaks': self._find_density_peaks(all_point_perp),
                'distribution_perplexity_density_peaks': self._find_density_peaks(all_dist_perp)
            }
        }
        
        analysis_results = {
            'global_statistics': global_stats,
            'trajectory_statistics': trajectory_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis_results
    
    def _find_density_peaks(self, data: np.ndarray, num_bins: int = 50) -> List[float]:
        """Find peaks in 1D density distribution"""
        try:
            hist, bin_edges = np.histogram(data, bins=num_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Find local maxima
            peaks = []
            for i in range(1, len(hist)-1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    peaks.append(float(bin_centers[i]))
            
            return sorted(peaks, key=lambda x: hist[np.argmin(np.abs(bin_centers - x))], reverse=True)[:3]  # Top 3 peaks
        except:
            return []
    
    def save_generation_data(self, trajectories: List[Dict], output_dir: str) -> None:
        """Save generation data when generation mode was used"""
        
        print(f"📝 Saving generation data...")
        
        generation_data = {
            'generation_info': {
                'num_generate_tokens': self.generate_tokens,
                'generation_temperature': 0.8,  # Fixed temperature we used
                'model_name': self.model_name,
                'base_model_name': self.base_model_name,
                'is_random_model': self.is_random_model,
                'total_trajectories': len(trajectories),
                'timestamp': datetime.now().isoformat()
            },
            'trajectories': []
        }
        
        for i, traj in enumerate(trajectories):
            if traj.get('generation_used', False):
                # Extract prompt and generated parts
                prompt_length = traj.get('prompt_length', 0)
                all_tokens = traj.get('token_sequence', [])
                
                prompt_tokens = all_tokens[:prompt_length]
                generated_tokens = all_tokens[prompt_length:]
                
                # Decode text
                prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)
                
                # Extract step data for generated tokens
                generated_steps = []
                for step_data in traj['steps']:
                    if step_data.get('is_generated', False):
                        generated_steps.append({
                            'step': step_data['step'],
                            'token': step_data['token'],
                            'momentum': step_data['momentum'],
                            'point_perplexity': step_data['point_perplexity'],
                            'distribution_perplexity': step_data['distribution_perplexity'],
                            'token_type': step_data['token_type']
                        })
                
                trajectory_data = {
                    'trajectory_id': i,
                    'original_text': traj['text'],  # Original Wikipedia text
                    'prompt': {
                        'text': prompt_text,
                        'tokens': prompt_tokens,
                        'length': len(prompt_tokens)
                    },
                    'generated': {
                        'text': generated_text,
                        'tokens': generated_tokens,
                        'length': len(generated_tokens),
                        'steps': generated_steps
                    },
                    'full_sequence': {
                        'text': full_text,
                        'tokens': all_tokens,
                        'total_length': len(all_tokens)
                    },
                    'trajectory_stats': {
                        'total_steps': len(traj['steps']),
                        'prompt_steps': prompt_length,
                        'generated_steps': len(generated_tokens)
                    }
                }
                
                generation_data['trajectories'].append(trajectory_data)
        
        # Save generation data
        generation_file = f"{output_dir}/generation_data.json"
        with open(generation_file, 'w') as f:
            json.dump(generation_data, f, indent=2, default=str)
        
        print(f"  ✅ Generation data saved to {generation_file}")
        print(f"  📊 Saved data for {len(generation_data['trajectories'])} generated trajectories")
        
        # Save a human-readable text file with generations
        text_file = f"{output_dir}/generated_texts.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("Generated Trajectories\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Generated tokens per trajectory: {self.generate_tokens}\n")
            f.write(f"Total trajectories: {len(generation_data['trajectories'])}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            for i, traj_data in enumerate(generation_data['trajectories']):
                f.write(f"Trajectory {i + 1}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Original Text: {traj_data['original_text'][:100]}...\n\n")
                f.write(f"PROMPT ({traj_data['prompt']['length']} tokens):\n")
                f.write(f"{traj_data['prompt']['text']}\n\n")
                f.write(f"GENERATED ({traj_data['generated']['length']} tokens):\n")
                f.write(f"{traj_data['generated']['text']}\n\n")
                f.write(f"FULL SEQUENCE:\n")
                f.write(f"{traj_data['full_sequence']['text']}\n\n")
                f.write("=" * 50 + "\n\n")
        
        print(f"  📄 Human-readable generations saved to {text_file}")

    def save_results(self, trajectories: List[Dict], analysis: Dict, output_dir: str) -> None:
        """Save all results and analysis"""
        
        print(f"💾 Saving phase space analysis results...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main analysis
        with open(f"{output_dir}/phase_space_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save trajectory data (without hidden states to save space)
        trajectories_export = []
        for traj in trajectories:
            traj_export = {
                'text': traj['text'],
                'token_sequence': traj['token_sequence'],
                'steps': [
                    {
                        'step': step['step'],
                        'momentum': step['momentum'],
                        'point_perplexity': step['point_perplexity'],
                        'distribution_perplexity': step['distribution_perplexity'],
                        'token': step['token']
                    }
                    for step in traj['steps']
                ],
                'momentum': traj['momentum'],
                'point_perplexity': traj['point_perplexity'],
                'distribution_perplexity': traj['distribution_perplexity']
            }
            trajectories_export.append(traj_export)
        
        with open(f"{output_dir}/trajectories.json", 'w') as f:
            json.dump(trajectories_export, f, indent=2, default=str)
        
        # Save raw data as numpy arrays for further analysis
        all_momentum = []
        all_point_perp = []
        all_dist_perp = []
        
        for traj in trajectories:
            all_momentum.extend(traj['momentum'][1:])  # Skip first step
            all_point_perp.extend(traj['point_perplexity'][1:])
            all_dist_perp.extend(traj['distribution_perplexity'][1:])
        
        np.save(f"{output_dir}/momentum_data.npy", np.array(all_momentum))
        np.save(f"{output_dir}/point_perplexity_data.npy", np.array(all_point_perp))
        np.save(f"{output_dir}/distribution_perplexity_data.npy", np.array(all_dist_perp))
        
        # Save generation data if generation mode was used
        if self.generate_tokens and self.generate_tokens > 0:
            self.save_generation_data(trajectories, output_dir)
        
        # Save comprehensive experiment configuration
        import sys
        import platform
        
        experiment_config = {
            'experiment_info': {
                'script_name': 'momentum_perplexity_phase_space_experiment.py',
                'experiment_type': 'Momentum-Perplexity Phase Space Analysis',
                'execution_timestamp': datetime.now().isoformat(),
                'command_line_args': sys.argv[1:] if len(sys.argv) > 1 else [],
                'python_version': platform.python_version(),
                'platform': platform.platform(),
                'working_directory': os.getcwd()
            },
            'model_configuration': {
                'model_name': self.model_name,
                'base_model_name': self.base_model_name,
                'is_random_model': self.is_random_model,
                'device': self.device,
                'hidden_size': self.model.config.hidden_size,
                'vocab_size': self.model.config.vocab_size,
                'tokenizer_info': {
                    'pad_token': self.tokenizer.pad_token,
                    'eos_token': self.tokenizer.eos_token,
                    'vocab_size': self.tokenizer.vocab_size
                }
            },
            'experiment_parameters': {
                'num_articles': len(trajectories),
                'max_steps_per_text': len(trajectories[0]['steps']) if trajectories else 0,
                'total_trajectories': len(trajectories),
                'total_phase_space_points': sum(len(traj['steps']) for traj in trajectories),
                'generation_mode': {
                    'enabled': self.generate_tokens is not None and self.generate_tokens > 0,
                    'num_generate_tokens': self.generate_tokens if self.generate_tokens else 0,
                    'generation_temperature': 0.8 if self.generate_tokens else None
                }
            },
            'execution_summary': {
                'trajectories_extracted': len(trajectories),
                'phase_space_points': sum(len(traj['steps']) for traj in trajectories),
                'momentum_range': {
                    'min': float(np.min(all_momentum)) if all_momentum else 0,
                    'max': float(np.max(all_momentum)) if all_momentum else 0,
                    'mean': float(np.mean(all_momentum)) if all_momentum else 0
                },
                'point_perplexity_range': {
                    'min': float(np.min(all_point_perp)) if all_point_perp else 0,
                    'max': float(np.max(all_point_perp)) if all_point_perp else 0,
                    'mean': float(np.mean(all_point_perp)) if all_point_perp else 0
                },
                'distribution_perplexity_range': {
                    'min': float(np.min(all_dist_perp)) if all_dist_perp else 0,
                    'max': float(np.max(all_dist_perp)) if all_dist_perp else 0,
                    'mean': float(np.mean(all_dist_perp)) if all_dist_perp else 0
                }
            }
        }
        
        with open(f"{output_dir}/experiment_config.json", 'w') as f:
            json.dump(experiment_config, f, indent=2, default=str)
        
        print(f"✅ Results saved to {output_dir}/")
    
    def run_phase_space_experiment(self, num_articles: int = 20, max_steps_per_text: int = 10, perplexity_cutoff: Optional[float] = None) -> Dict:
        """Run the complete phase space analysis experiment"""
        
        print("🌌 MOMENTUM-PERPLEXITY PHASE SPACE EXPERIMENT")
        print("=" * 70)
        
        # Load text samples
        text_samples = self.load_wikipedia_samples(num_articles)
        
        # Extract trajectories
        trajectories = self.extract_all_trajectories(text_samples, max_steps_per_text)
        
        if not trajectories:
            raise RuntimeError("No trajectories extracted. Cannot proceed with analysis.")
        
        # Analyze phase space statistics
        analysis = self.analyze_phase_space_statistics(trajectories)
        
        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"phase_space_analysis_{timestamp}"
        
        self.create_phase_space_visualizations(trajectories, output_dir, perplexity_cutoff)
        
        # Save results
        self.save_results(trajectories, analysis, output_dir)
        
        # Print summary
        print(f"\n🎉 PHASE SPACE ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {output_dir}/")
        print(f"🎯 Trajectories analyzed: {len(trajectories)}")
        print(f"📊 Phase space points: {analysis['global_statistics']['total_phase_space_points']}")
        
        global_corr = analysis['global_statistics']['global_correlations']
        print(f"\n🔗 Global Phase Space Correlations:")
        print(f"   Momentum ↔ Point Perplexity: {global_corr['momentum_vs_point_perplexity']:.3f}")
        print(f"   Momentum ↔ Distribution Perplexity: {global_corr['momentum_vs_distribution_perplexity']:.3f}")
        print(f"   Point ↔ Distribution Perplexity: {global_corr['point_vs_distribution_perplexity']:.3f}")
        
        return {
            'trajectories': trajectories,
            'analysis': analysis,
            'output_dir': output_dir
        }


def main():
    """Run the phase space analysis experiment"""
    parser = argparse.ArgumentParser(description="Momentum-Perplexity Phase Space Analysis")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model to use (add ':random' suffix for random-weight baseline, e.g., 'Qwen/Qwen2.5-1.5B-Instruct:random')")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--num_articles", type=int, default=20, help="Number of Wikipedia articles to analyze")
    parser.add_argument("--max_steps_per_text", type=int, default=10, help="Max trajectory steps per text")
    parser.add_argument("--perplexity_cutoff", type=float, default=None, help="Maximum perplexity value for visualization")
    parser.add_argument("--generate_tokens", type=int, default=None, help="Number of tokens to generate manually per trajectory (if not specified, uses existing text only)")
    
    args = parser.parse_args()

    print("🚀 Starting Momentum-Perplexity Phase Space Analysis")
    
    # Initialize experiment
    experiment = MomentumPerplexityPhaseSpaceExperiment(
        model_name=args.model_name,
        device=args.device,
        generate_tokens=args.generate_tokens
    )
    
    # Run experiment
    results = experiment.run_phase_space_experiment(
        num_articles=args.num_articles,
        max_steps_per_text=args.max_steps_per_text,
        perplexity_cutoff=args.perplexity_cutoff
    )
    
    print("\n🌌 Phase space analysis reveals the hidden dynamics of language model generation!")


if __name__ == "__main__":
    main() 