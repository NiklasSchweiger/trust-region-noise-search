"""Trust region state management.

This module provides the _TRState class for tracking individual trust region state,
including adaptive radius management, success/failure counters, and restart logic.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .tr_utils import calculate_1_5_rule_length


@dataclass
class TRState:
    """Trust region state tracker for adaptive radius management.
    
    Tracks the state of a single trust region, including its size, success/failure
    counters, and adaptive adjustment logic. The trust region expands after consecutive
    successes and shrinks after consecutive failures.
    
    Attributes:
        update_factor: Factor by which to update the trust region length after consecutive successes
        length: Current trust region side length
        min_length: Minimum allowed trust region length
        max_length: Maximum allowed trust region length
        prob_perturb: Probability of perturbing each dimension (for sampling-based mode)
        min_prob_perturb: Minimum perturbation probability
        max_prob_perturb: Maximum perturbation probability
        max_comb: Maximum combined value of length + prob_perturb (for sampling-based mode)
        min_comb: Minimum combined value of length + prob_perturb (for sampling-based mode)
        success_tolerance: Number of consecutive successes before expanding
        failure_tolerance: Number of consecutive failures before shrinking
        success_counter: Current count of consecutive successes
        failure_counter: Current count of consecutive failures
        restart_triggered: Flag indicating trust region should restart at global best
        best_value: Best objective value found in this trust region
        sampling_based: If True, randomly sample length and prob_perturb
        device: Device for tensor operations
        update_mode: Update mode - "standard" or "one_fifth"
    """
    update_factor: float = 2.0
    length: float = 0.8
    min_length: float = 0.1
    max_length: float = 1.6
    prob_perturb: float = 0.5
    min_prob_perturb: float = 0.1
    max_prob_perturb: float = 0.9
    max_comb: float = 2.6
    min_comb: float = 0.2
    success_tolerance: int = 3
    failure_tolerance: int = 3
    success_counter: int = 0
    failure_counter: int = 0
    restart_triggered: bool = False
    best_value: float = float("-inf")
    sampling_based: bool = False
    device: str = "cuda"
    update_mode: str = "standard"  # "standard" | "one_fifth"

    def update(self, new_value: float) -> None:
        """Update trust region state based on new observation."""
        improved = new_value > self.best_value + 1e-12
        
        if improved:
            self.best_value = new_value
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.update_mode == "one_fifth":
            # Approximate recent success rate using counters and apply 1/5th rule.
            total = self.success_counter + self.failure_counter
            success_rate = float(self.success_counter) / float(total) if total > 0 else 0.0
            self.length = calculate_1_5_rule_length(
                current_length=self.length,
                success_rate=success_rate,
                min_length=self.min_length,
                max_length=self.max_length,
            )
            # Reset counters after an adaptation step to avoid over-counting old history.
            self.success_counter = 0
            self.failure_counter = 0
        else:
            # Standard TuRBO-style rule: expand/shrink after consecutive successes/failures.
            if self.success_counter >= self.success_tolerance:
                self.length = min(self.length * self.update_factor, self.max_length)
                self.success_counter = 0

            if self.failure_counter >= self.failure_tolerance:
                new_len = max(self.length / self.update_factor, self.min_length)
                if new_len == self.min_length and self.length == self.min_length:
                    self.restart_triggered = True
                self.length = new_len
                self.failure_counter = 0
        #print(f"[TRS] Length update: {self.length}")

        # Sampling-based mode: randomly sample length and prob_perturb
        if self.sampling_based:
            # Sample length and prob_perturb uniformly within their ranges
            self.length = float(
                torch.rand(1, device=self.device).item() * 
                (self.max_length - self.min_length) + self.min_length
            )
            self.prob_perturb = float(
                torch.rand(1, device=self.device).item() * 
                (self.max_prob_perturb - self.min_prob_perturb) + self.min_prob_perturb
            )
            # Resample if combined value is outside allowed bounds
            while (self.length + self.prob_perturb > self.max_comb or 
                   self.length + self.prob_perturb < self.min_comb):
                self.length = float(
                    torch.rand(1, device=self.device).item() * 
                    (self.max_length - self.min_length) + self.min_length
                )
                self.prob_perturb = float(
                    torch.rand(1, device=self.device).item() * 
                    (self.max_prob_perturb - self.min_prob_perturb) + self.min_prob_perturb
                )
        
        # # Final Check: ensure length and prob_perturb are within bounds
        if self.prob_perturb >= 0.3 and self.length >= 2.0:
            self.prob_perturb = 0.2
        if self.prob_perturb >= 0.5 and self.length >= 2.0:
            self.prob_perturb = 0.2
        if self.prob_perturb >= 0.7 and self.length >= 1.6:
            self.prob_perturb = 0.5
        if self.prob_perturb >= 0.9 and self.length >= 1.2:
            self.prob_perturb = 0.7

