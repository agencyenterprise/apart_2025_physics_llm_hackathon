#!/usr/bin/env python3
"""
Single Direction Steering
=========================

Finds the best single direction using random sampling and applies it.
"""

import torch
import numpy as np
from typing import Dict, Any
from .base_steering import BaseSteering


class SingleDirectionSteering(BaseSteering):
    """Single direction steering using random sampling."""
    
    def __init__(self, model, tokenizer, device, num_directions: int = 500):
        """Initialize single direction steering.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            device: The device to use
            num_directions: Number of random directions to sample
        """
        super().__init__(model, tokenizer, device)
        self.num_directions = num_directions
    
    def steer(self, baseline_hidden: torch.Tensor, target_token_id: int, 
              baseline_probs: torch.Tensor) -> Dict[str, Any]:
        """Apply single direction steering."""
        print(f"   Finding best single direction...")
        
        # Find best direction using random sampling
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
        
        # Get best direction
        best_idx = np.argmax(direction_effects)
        best_direction = random_directions[best_idx]
        best_effect = direction_effects[best_idx]
        
        print(f"   Best direction effect: {best_effect:.6f}")
        
        # Test magnitudes
        magnitude_result = self.test_magnitudes(baseline_hidden, best_direction, 
                                              target_token_id, baseline_probs)
        
        return {
            'method_name': 'single_direction',
            'direction_effect': best_effect,
            'best_direction': best_direction,
            'num_directions_tested': self.num_directions,
            **magnitude_result
        } 