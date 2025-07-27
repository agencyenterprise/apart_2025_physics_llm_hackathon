#!/usr/bin/env python3
"""
Base Steering Class
==================

Common interface and utilities for all steering approaches.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F


class BaseSteering(ABC):
    """Base class for all steering approaches."""
    
    def __init__(self, model, tokenizer, device):
        """Initialize the steering approach.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            device: The device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hidden_dim = model.config.hidden_size
        self.vocab_size = len(tokenizer)
    
    @abstractmethod
    def steer(self, baseline_hidden: torch.Tensor, target_token_id: int, 
              baseline_probs: torch.Tensor) -> Dict[str, Any]:
        """Apply steering to make target token more probable.
        
        Args:
            baseline_hidden: The baseline hidden state
            target_token_id: ID of the target token
            baseline_probs: Baseline probability distribution
            
        Returns:
            Dictionary with steering results
        """
        pass
    
    def get_probs_from_hidden(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Get probability distribution from hidden state."""
        with torch.no_grad():
            logits = self.model.lm_head(hidden_state)
            probs = F.softmax(logits, dim=-1)
        return probs
    
    def get_token_rank(self, probs: torch.Tensor, token_id: int) -> int:
        """Get rank of token in probability distribution."""
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1
        return rank
    
    def test_magnitudes(self, baseline_hidden: torch.Tensor, direction: torch.Tensor,
                       target_token_id: int, baseline_probs: torch.Tensor,
                       magnitude_range: list = None) -> Dict:
        """Test different magnitudes for a given direction."""
        if magnitude_range is None:
            magnitude_range = [10, 25, 50, 75, 100, 150, 200, 300, 500]
            
        best_magnitude = None
        best_prob = baseline_probs[target_token_id].item()
        best_rank = self.get_token_rank(baseline_probs, target_token_id)
        achieved_top_1 = False
        
        for magnitude in magnitude_range:
            steered_hidden = baseline_hidden + magnitude * direction
            new_probs = self.get_probs_from_hidden(steered_hidden)
            new_prob = new_probs[target_token_id].item()
            new_rank = self.get_token_rank(new_probs, target_token_id)
            
            if new_rank == 1 and not achieved_top_1:
                # First time achieving top-1
                best_magnitude = magnitude
                best_prob = new_prob
                best_rank = new_rank
                achieved_top_1 = True
                break
            elif new_prob > best_prob:
                # Better probability (but not necessarily top-1)
                best_magnitude = magnitude
                best_prob = new_prob
                best_rank = new_rank
        
        return {
            'best_magnitude': best_magnitude,
            'final_prob': best_prob,
            'final_rank': best_rank,
            'achieved_top_1': achieved_top_1
        }
    
    def evaluate_steering(self, baseline_hidden: torch.Tensor, direction: torch.Tensor,
                         magnitude: float, target_token_id: int) -> Dict:
        """Evaluate steering with given direction and magnitude."""
        # Apply steering
        steered_hidden = baseline_hidden + magnitude * direction
        
        # Get probabilities
        baseline_probs = self.get_probs_from_hidden(baseline_hidden)
        steered_probs = self.get_probs_from_hidden(steered_hidden)
        
        # Analyze results
        baseline_target_prob = baseline_probs[target_token_id].item()
        steered_target_prob = steered_probs[target_token_id].item()
        baseline_target_rank = self.get_token_rank(baseline_probs, target_token_id)
        steered_target_rank = self.get_token_rank(steered_probs, target_token_id)
        
        prob_improvement = steered_target_prob - baseline_target_prob
        rank_improvement = baseline_target_rank - steered_target_rank
        achieved_top_1 = (steered_target_rank == 1)
        
        return {
            'baseline_target_prob': baseline_target_prob,
            'steered_target_prob': steered_target_prob,
            'prob_improvement': prob_improvement,
            'baseline_target_rank': baseline_target_rank,
            'steered_target_rank': steered_target_rank,
            'rank_improvement': rank_improvement,
            'achieved_top_1': achieved_top_1,
            'final_prob': steered_target_prob,
            'final_rank': steered_target_rank
        } 