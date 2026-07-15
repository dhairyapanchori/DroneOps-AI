"""In-memory, per-episode training metrics (console summary at end of run)."""


class Metrics:
    """
    Tracks per-episode stats across the full training run.
    """

    def __init__(self):
        self.rewards         = []
        self.targets_hit     = []
        self.drones_failed   = []
        self.coordination    = []   # avg unique target coverage per episode
        self.actor_losses    = []
        self.critic_losses   = []
        self.evolution_gains = []   # score improvements from evolution passes

    def log(self, reward, targets_hit=0, drones_failed=0,
            coordination=0.0, a_loss=None, c_loss=None):
        self.rewards.append(reward)
        self.targets_hit.append(targets_hit)
        self.drones_failed.append(drones_failed)
        self.coordination.append(coordination)
        if a_loss is not None:
            self.actor_losses.append(a_loss)
        if c_loss is not None:
            self.critic_losses.append(c_loss)

    def log_evolution(self, gain):
        self.evolution_gains.append(gain)

    def summary(self):
        if not self.rewards:
            return {}

        def avg(lst):
            return round(sum(lst) / len(lst), 3) if lst else 0.0

        return {
            "episodes"          : len(self.rewards),
            "mean_reward"       : avg(self.rewards),
            "best_reward"       : round(max(self.rewards), 3),
            "last_reward"       : round(self.rewards[-1], 3),
            "mean_targets_hit"  : avg(self.targets_hit),
            "mean_failures"     : avg(self.drones_failed),
            "mean_coordination" : avg(self.coordination),
            "evolution_passes"  : len(self.evolution_gains),
            "evolution_gains"   : self.evolution_gains,
        }
