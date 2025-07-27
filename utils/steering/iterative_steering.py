#!/usr/bin/env python3
"""
Iterative Steering
=================

Applies steering iteratively, refining the direction at each step.
"""

import torch
import numpy as np
from typing import Dict, Any, List
from .base_steering import BaseSteering


class IterativeSteering(BaseSteering):
    """Iterative steering that refines direction at each step."""
    
    def __init__(self, model, tokenizer, device, max_iterations: int = 5, 
                 directions_per_iter: int = 200, step_magnitude: float = 50.0):
        """Initialize iterative steering.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            device: The device to use
            max_iterations: Maximum number of iterations
            directions_per_iter: Number of directions to test per iteration
            step_magnitude: Magnitude of each steering step
        """
        super().__init__(model, tokenizer, device)
        self.max_iterations = max_iterations
        self.directions_per_iter = directions_per_iter
        self.step_magnitude = step_magnitude
    
    def steer(self, baseline_hidden: torch.Tensor, target_token_id: int, 
              baseline_probs: torch.Tensor) -> Dict[str, Any]:
        """Apply iterative steering."""
        print(f"   Iterative steering with {self.max_iterations} steps...")
        
        current_hidden = baseline_hidden.clone()
        iteration_results = []
        
        for iteration in range(self.max_iterations):
            print(f"     Iteration {iteration + 1}/{self.max_iterations}")
            
            # Get current probabilities
            current_probs = self.get_probs_from_hidden(current_hidden)
            current_prob = current_probs[target_token_id].item()
            current_rank = self.get_token_rank(current_probs, target_token_id)
            
            print(f"       Current: prob={current_prob:.6f}, rank={current_rank}")
            
            # If we achieved top-1, stop
            if current_rank == 1:
                print(f"       ✅ Achieved top-1 at iteration {iteration + 1}")
                break
            
            # Find best direction from current state
            direction_effects = []
            random_directions = []
            
            for i in range(self.directions_per_iter):
                direction = torch.randn(self.hidden_dim, device=self.device)
                direction = direction / torch.norm(direction)
                random_directions.append(direction)
                
                # Test small perturbation
                perturbed_hidden = current_hidden + 1.0 * direction
                new_probs = self.get_probs_from_hidden(perturbed_hidden)
                prob_change = new_probs[target_token_id].item() - current_probs[target_token_id].item()
                direction_effects.append(prob_change)
            
            # Get best direction for this iteration
            best_idx = np.argmax(direction_effects)
            best_direction = random_directions[best_idx]
            best_effect = direction_effects[best_idx]
            
            # Apply steering step
            current_hidden = current_hidden + self.step_magnitude * best_direction
            
            # Record iteration results
            post_step_probs = self.get_probs_from_hidden(current_hidden)
            iteration_results.append({
                'iteration': iteration + 1,
                'direction_effect': best_effect,
                'step_magnitude': self.step_magnitude,
                'prob_before': current_prob,
                'rank_before': current_rank,
                'prob_after': post_step_probs[target_token_id].item(),
                'rank_after': self.get_token_rank(post_step_probs, target_token_id)
            })
        
        # Final state
        final_probs = self.get_probs_from_hidden(current_hidden)
        final_prob = final_probs[target_token_id].item()
        final_rank = self.get_token_rank(final_probs, target_token_id)
        
        return {
            'method_name': 'iterative',
            'iterations_completed': len(iteration_results),
            'iteration_results': iteration_results,
            'final_prob': final_prob,
            'final_rank': final_rank,
            'achieved_top_1': (final_rank == 1),
            'final_hidden': current_hidden
        } 