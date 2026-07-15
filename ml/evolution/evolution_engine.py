import torch
import numpy as np
import copy


class EvolutionEngine:
    """
    Periodic evolutionary improvement on top of gradient-based training (SAC).

    Role in training:
        SAC updates the actor every few steps via gradient descent.
        Every EVOLVE_EVERY episodes, the EvolutionEngine runs a separate
        evaluation pass: it creates mutated copies of the current actor,
        rolls each one out in the environment, and if any mutant scores
        higher than the current actor it replaces it.

        This helps escape local optima that gradient descent gets stuck in,
        especially early in training when the critic is still unreliable.

    Strategy: (1 + pop_size) ES — elitist selection, current actor always
    competes against its mutants.
    """

    def __init__(self, actor, pop_size=8, sigma=0.03):
        self.actor    = actor
        self.pop_size = pop_size
        self.sigma    = sigma

    def _mutate(self, actor):
        """Return a deep copy of `actor` with Gaussian noise (std=sigma) added."""
        mutant = copy.deepcopy(actor)
        with torch.no_grad():
            for p in mutant.parameters():
                p.add_(torch.randn_like(p) * self.sigma)
        return mutant

    def evolve(self, fitness_fn):
        """
        Args:
            fitness_fn: callable(actor) -> float
                        Should roll out the actor for one episode and
                        return the total reward.
        Returns:
            (best_actor, best_score, improved: bool)
        """
        # Score the current actor (elitist baseline)
        current_score = fitness_fn(self.actor)

        # Generate and score mutants
        population = [self._mutate(self.actor) for _ in range(self.pop_size)]
        scores     = [fitness_fn(a) for a in population]

        best_idx   = int(np.argmax(scores))
        best_score = scores[best_idx]

        improved = False
        if best_score > current_score:
            self.actor = population[best_idx]
            improved   = True

        return self.actor, max(best_score, current_score), improved
