#!/usr/bin/env python3
"""
Jacobian Path Search Steering
============================

Uses closed-form Jacobian computation and beam search through hidden space.

This is the most advanced approach that uses exact gradient computation:
∇_h log p_t = W_t - Σ_j p_j W_j

Performs beam search through hidden space with analytical steering directions.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_steering import BaseSteering


class JacobianPathSearch(BaseSteering):
    """Closed-form Jacobian path search steering."""
    
    def __init__(self, model, tokenizer, device, beam_width: int = 5, min_steps: int = 3, 
                 max_steps: int = 8, top_k_candidates: int = 10, alpha_max: float = 200.0):
        """Initialize Jacobian path search steering.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            device: The device to use
            beam_width: Number of paths to maintain in beam search
            min_steps: Minimum steering steps even if target achieved early
            max_steps: Maximum search depth
            top_k_candidates: Top-K tokens to consider at each step
            alpha_max: Maximum step size
        """
        super().__init__(model, tokenizer, device)
        self.beam_width = beam_width
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.top_k_candidates = top_k_candidates
        self.alpha_max = alpha_max
    
    def compute_closed_form_gradient(self, hidden_state: torch.Tensor, 
                                   target_token_id: int) -> Tuple[torch.Tensor, float]:
        """Compute exact gradient using closed-form Jacobian.
        
        Returns:
            Tuple of (steering_direction, grad_norm)
        """
        # Get LM head weights and current probabilities
        lm_head_weights = self.model.lm_head.weight  # [vocab_size, hidden_dim]
        current_probs = self.get_probs_from_hidden(hidden_state)
        
        # Closed-form gradient: ∇_h log p_t = W_t - Σ_j p_j W_j
        W_t = lm_head_weights[target_token_id]  # Weight row for target token
        weighted_avg = torch.sum(current_probs.unsqueeze(1) * lm_head_weights, dim=0)  # Σ_j p_j W_j
        grad_log_p = W_t - weighted_avg  # Exact gradient
        
        # Get gradient norm and normalized direction
        grad_norm = torch.norm(grad_log_p)
        if grad_norm < 1e-8:
            return None, 0.0
        
        steering_direction = grad_log_p / grad_norm
        return steering_direction, grad_norm.item()
    
    def compute_optimal_step_size(self, hidden_state: torch.Tensor, target_token_id: int,
                                grad_norm: float, token_id: int) -> float:
        """Compute optimal step size using margin-based inversion."""
        current_logits = self.model.lm_head(hidden_state)
        
        if token_id == target_token_id:
            # For target token: want to increase its logit above others
            max_other_logit = torch.max(current_logits[current_logits != current_logits[target_token_id]])
            margin = max_other_logit - current_logits[target_token_id] + 2.0  # +2 safety margin
            alpha = torch.clamp(margin / grad_norm, -self.alpha_max, self.alpha_max).item()
        else:
            # For other tokens: smaller step toward improving overall distribution
            alpha = min(50.0, self.alpha_max / 4)  # More conservative for non-target
        
        return alpha
    
    def steer(self, baseline_hidden: torch.Tensor, target_token_id: int, 
              baseline_probs: torch.Tensor, debug: bool = True) -> Dict[str, Any]:
        """Apply Jacobian path search steering."""
        if debug:
            print(f"   🧮 Closed-form Jacobian path search (min_steps={self.min_steps}, max_steps={self.max_steps})...")
        
        # Initialize beam: (hidden_state, path_score, steps_taken, token_sequence)
        beam = [(baseline_hidden.clone(), 0.0, 0, [])]
        best_result = None
        best_score = -float('inf')
        search_complete = False  # Flag to stop entire search
        
        for step in range(self.max_steps):
            if search_complete:
                break
            if debug:
                print(f"     🔍 Step {step + 1}/{self.max_steps}, beam size: {len(beam)}")
            
            new_beam = []
            
            for hidden_state, current_score, steps_taken, token_sequence in beam:
                # Get current probabilities
                current_probs = self.get_probs_from_hidden(hidden_state)
                
                # Check if target is already top-1
                current_target_rank = self.get_token_rank(current_probs, target_token_id)
                if current_target_rank == 1 and steps_taken >= self.min_steps:
                    # Found solution and met minimum steps! Stop entire search
                    result_score = current_score + 1000.0  # Bonus for achieving top-1
                    best_result = {
                        'final_hidden': hidden_state,
                        'final_prob': current_probs[target_token_id].item(),
                        'final_rank': 1,
                        'achieved_top_1': True,
                        'steps_taken': steps_taken,
                        'path_score': result_score,
                        'token_sequence': token_sequence,
                        'search_terminated': 'target_achieved_min_steps'
                    }
                    # Stop entire search - we found the solution at min_steps
                    search_complete = True
                    break
                elif current_target_rank == 1 and steps_taken < self.min_steps:
                    # Found rank 1 but haven't reached min_steps yet - keep this result but continue
                    result_score = current_score + 500.0  # Partial bonus for early success
                    if result_score > best_score:
                        best_score = result_score
                        best_result = {
                            'final_hidden': hidden_state,
                            'final_prob': current_probs[target_token_id].item(),
                            'final_rank': 1,
                            'achieved_top_1': True,
                            'steps_taken': steps_taken,
                            'path_score': result_score,
                            'token_sequence': token_sequence,
                            'search_terminated': 'early_success_continuing'
                        }
                    # @FIXME: ADDING TEMPORARY TERMINATION CONDITION
                    search_complete = True
                    break
                    # Continue expanding this path until min_steps
                
                # Get top-K candidate tokens to consider
                top_probs, top_indices = torch.topk(current_probs, self.top_k_candidates)
                
                # Add target token if not in top-K
                candidates = list(top_indices.cpu().numpy())
                if target_token_id not in candidates:
                    candidates.append(target_token_id)
                
                # For each candidate token, compute optimal steering direction
                for token_id in candidates:
                    # Compute closed-form gradient
                    steering_direction, grad_norm = self.compute_closed_form_gradient(
                        hidden_state, token_id
                    )
                    
                    if steering_direction is None:
                        continue  # Skip if gradient is too small
                    
                    # Compute optimal step size
                    alpha = self.compute_optimal_step_size(
                        hidden_state, target_token_id, grad_norm, token_id
                    )
                    
                    if abs(alpha) < 1.0:  # Skip very small steps
                        continue
                    
                    # Apply steering step
                    new_hidden = hidden_state + alpha * steering_direction
                    
                    # Evaluate new state
                    new_probs = self.get_probs_from_hidden(new_hidden)
                    new_target_prob = new_probs[target_token_id].item()
                    new_target_rank = self.get_token_rank(new_probs, target_token_id)
                    
                    # Compute step reward
                    prob_improvement = new_target_prob - current_probs[target_token_id].item()
                    rank_improvement = current_target_rank - new_target_rank
                    
                    # Reward function: prioritize probability increase and rank improvement
                    step_reward = prob_improvement * 10.0 + max(0, rank_improvement) * 0.1
                    
                    # Penalize too many steps (encourage efficiency)
                    step_penalty = steps_taken * 0.01
                    
                    new_score = current_score + step_reward - step_penalty
                    new_token_sequence = token_sequence + [token_id]
                    
                    # Add to new beam  
                    new_beam.append((new_hidden, new_score, steps_taken + 1, new_token_sequence))
            
            if not new_beam:
                if debug:
                    print(f"       ⚠️  Beam search terminated: no valid moves")
                break
            
            # Sort by score and keep top beam_width paths
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:self.beam_width]
            
            # Track best path found so far
            best_in_beam = beam[0]
            beam_hidden, beam_score, beam_steps, beam_sequence = best_in_beam
            beam_probs = self.get_probs_from_hidden(beam_hidden)
            beam_target_prob = beam_probs[target_token_id].item()
            beam_target_rank = self.get_token_rank(beam_probs, target_token_id)
            
            if debug:
                print(f"       🏆 Best in beam: prob={beam_target_prob:.6f}, rank={beam_target_rank}, score={beam_score:.3f}")
            
            # Update best result if this is better
            if beam_score > best_score:
                best_score = beam_score
                # Determine termination reason
                if step == self.max_steps - 1:
                    if beam_steps >= self.min_steps:
                        termination_reason = 'max_steps_min_met'
                    else:
                        termination_reason = 'max_steps_min_not_met'
                else:
                    termination_reason = 'in_progress'
                
                best_result = {
                    'final_hidden': beam_hidden,
                    'final_prob': beam_target_prob,
                    'final_rank': beam_target_rank,
                    'achieved_top_1': (beam_target_rank == 1),
                    'steps_taken': beam_steps,
                    'path_score': beam_score,
                    'token_sequence': beam_sequence,
                    'search_terminated': termination_reason
                }
        
        # Return best result found
        if best_result is None:
            # Fallback if nothing worked
            baseline_target_prob = baseline_probs[target_token_id].item()
            baseline_target_rank = self.get_token_rank(baseline_probs, target_token_id)
            best_result = {
                'final_prob': baseline_target_prob,
                'final_rank': baseline_target_rank,
                'achieved_top_1': False,
                'steps_taken': 0,
                'path_score': 0.0,
                'token_sequence': [],
                'search_terminated': 'failed',
            }
        
        #if debug:
        print(f"   ✅ Path search completed:")
        print(f"      Final prob: {best_result['final_prob']:.6f}")
        print(f"      Final rank: {best_result['final_rank']}")
        print(f"      Steps taken: {best_result['steps_taken']}")
        print(f"      Success: {'✅' if best_result['achieved_top_1'] else '❌'}")
        
        # Add method name for consistency
        best_result['method_name'] = 'jacobian_path_search'
        
        return best_result 