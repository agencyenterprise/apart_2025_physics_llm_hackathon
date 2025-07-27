#!/usr/bin/env python3
"""
Orthogonal Steering
==================

Finds orthogonal beneficial directions and combines them.
"""

import torch
import numpy as np
from typing import Dict, Any, List
from .base_steering import BaseSteering


class OrthogonalSteering(BaseSteering):
    """Orthogonal steering that finds independent beneficial directions."""
    
    def __init__(self, model, tokenizer, device, max_directions: int = 4, 
                 directions_per_search: int = 300, ortho_directions_per_search: int = 200):
        """Initialize orthogonal steering.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            device: The device to use
            max_directions: Maximum number of orthogonal directions to find
            directions_per_search: Number of directions for initial search
            ortho_directions_per_search: Number of directions per orthogonal search
        """
        super().__init__(model, tokenizer, device)
        self.max_directions = max_directions
        self.directions_per_search = directions_per_search
        self.ortho_directions_per_search = ortho_directions_per_search
    
    def steer(self, baseline_hidden: torch.Tensor, target_token_id: int, 
              baseline_probs: torch.Tensor) -> Dict[str, Any]:
        """Apply orthogonal steering."""
        print(f"   Finding orthogonal beneficial directions...")
        
        # Find first best direction
        direction_effects = []
        random_directions = []
        
        for i in range(self.directions_per_search):
            direction = torch.randn(self.hidden_dim, device=self.device)
            direction = direction / torch.norm(direction)
            random_directions.append(direction)
            
            perturbed_hidden = baseline_hidden + 1.0 * direction
            new_probs = self.get_probs_from_hidden(perturbed_hidden)
            prob_change = new_probs[target_token_id].item() - baseline_probs[target_token_id].item()
            direction_effects.append(prob_change)
        
        # Get best direction
        best_idx = np.argmax(direction_effects)
        first_direction = random_directions[best_idx]
        first_effect = direction_effects[best_idx]
        
        print(f"   First direction effect: {first_effect:.6f}")
        
        # Find orthogonal directions
        orthogonal_directions = [first_direction]
        orthogonal_effects = [first_effect]
        
        for direction_num in range(2, self.max_directions + 1):
            # Generate directions orthogonal to existing ones
            best_ortho_effect = -float('inf')
            best_ortho_direction = None
            
            for i in range(self.ortho_directions_per_search):
                # Start with random direction
                direction = torch.randn(self.hidden_dim, device=self.device)
                
                # Make it orthogonal to existing directions using Gram-Schmidt
                for existing_dir in orthogonal_directions:
                    direction = direction - torch.dot(direction, existing_dir) * existing_dir
                
                # Normalize
                direction_norm = torch.norm(direction)
                if direction_norm > 1e-6:
                    direction = direction / direction_norm
                    
                    # Test this orthogonal direction
                    perturbed_hidden = baseline_hidden + 1.0 * direction
                    new_probs = self.get_probs_from_hidden(perturbed_hidden)
                    prob_change = new_probs[target_token_id].item() - baseline_probs[target_token_id].item()
                    
                    if prob_change > best_ortho_effect:
                        best_ortho_effect = prob_change
                        best_ortho_direction = direction
            
            if best_ortho_direction is not None and best_ortho_effect > 0:
                orthogonal_directions.append(best_ortho_direction)
                orthogonal_effects.append(best_ortho_effect)
                print(f"   Orthogonal direction {len(orthogonal_directions)} effect: {best_ortho_effect:.6f}")
            else:
                print(f"   No more beneficial orthogonal directions found")
                break
        
        # Combine orthogonal directions
        combined_direction = torch.zeros_like(baseline_hidden)
        for direction, effect in zip(orthogonal_directions, orthogonal_effects):
            combined_direction += effect * direction
        
        # Normalize combined direction
        combined_direction = combined_direction / torch.norm(combined_direction)
        
        # Test the combined orthogonal direction
        magnitude_result = self.test_magnitudes(baseline_hidden, combined_direction, 
                                              target_token_id, baseline_probs)
        
        return {
            'method_name': 'orthogonal',
            'num_orthogonal_directions': len(orthogonal_directions),
            'individual_effects': orthogonal_effects,
            'combined_direction': combined_direction,
            **magnitude_result
        } 