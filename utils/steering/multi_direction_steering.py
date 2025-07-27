#!/usr/bin/env python3
"""
Multi-Direction Steering  
========================

Combines multiple beneficial directions for improved steering.
"""

import torch
import numpy as np
from typing import Dict, Any
from .base_steering import BaseSteering


class MultiDirectionSteering(BaseSteering):
    """Multi-direction steering that combines top-K directions."""
    
    def __init__(self, model, tokenizer, device, num_directions: int = 500, top_k: int = 10):
        """Initialize multi-direction steering.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            device: The device to use
            num_directions: Number of random directions to sample
            top_k: Number of top directions to combine
        """
        super().__init__(model, tokenizer, device)
        self.num_directions = num_directions
        self.top_k = top_k
    
    def steer(self, baseline_hidden: torch.Tensor, target_token_id: int, 
              baseline_probs: torch.Tensor) -> Dict[str, Any]:
        """Apply multi-direction steering."""
        print(f"   Finding top {self.top_k} directions...")
        
        # Find multiple good directions
        direction_effects = []
        random_directions = []
        
        for i in range(self.num_directions):
            direction = torch.randn(self.hidden_dim, device=self.device)
            direction = direction / torch.norm(direction)
            random_directions.append(direction)
            
            # Test small perturbation
            perturbed_hidden = baseline_hidden + 1.0 * direction
            new_probs = self.get_probs_from_hidden(perturbed_hidden)
            prob_change = new_probs[target_token_id].item() - baseline_probs[target_token_id].item()
            direction_effects.append(prob_change)
        
        # Get top-K directions
        top_indices = np.argsort(direction_effects)[-self.top_k:]  # Top K
        top_directions = [random_directions[i] for i in top_indices]
        top_effects = [direction_effects[i] for i in top_indices]
        
        print(f"   Top effects: {[f'{e:.6f}' for e in top_effects[-3:]]}")
        
        # Combine directions (weighted by their effects)
        combined_direction = torch.zeros_like(baseline_hidden)
        total_weight = 0
        
        for direction, effect in zip(top_directions, top_effects):
            if effect > 0:  # Only use beneficial directions
                weight = max(0, effect)  # Use effect magnitude as weight
                combined_direction += weight * direction
                total_weight += weight
        
        # Normalize combined direction
        if total_weight > 0:
            combined_direction = combined_direction / total_weight
            combined_direction = combined_direction / torch.norm(combined_direction)
        else:
            # Fallback to best single direction
            best_idx = np.argmax(direction_effects)
            combined_direction = random_directions[best_idx]
        
        # Test the combined direction
        magnitude_result = self.test_magnitudes(baseline_hidden, combined_direction, 
                                              target_token_id, baseline_probs)
        
        num_beneficial = len([e for e in top_effects if e > 0])
        combined_effect = sum([e for e in top_effects if e > 0])
        
        return {
            'method_name': 'multi_direction',
            'num_directions_combined': num_beneficial,
            'combined_effect': combined_effect,
            'top_effects': top_effects,
            'combined_direction': combined_direction,
            **magnitude_result
        } 