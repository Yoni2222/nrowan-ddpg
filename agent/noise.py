from collections import deque
import numpy as np


class OnlineWeightAdjuster:
    """
    NROWAN Online Weight Adjustment.

    Controls xi, the weight of the noise-reduction loss D in the actor's
    objective:

        actor_loss = -Q(s, a) + xi * D

    Intuition (from NROWAN): when the agent performs well and consistently, we
    want it to REDUCE its exploration noise (act more deterministically). So xi
    grows as recent performance approaches the best seen so far, and shrinks
    when performance is poor (keep exploring).

    Performance is normalized online against the running min/max of the average
    episode return, so no prior knowledge of the reward scale is required.
    """

    def __init__(self, xi_max=1.0, window=20):
        self.xi_max = xi_max          # upper bound on the noise-reduction weight
        self.window = window
        self.recent = deque(maxlen=window)
        self.r_low = None
        self.r_high = None
        self.xi = 0.0

    def update(self, episode_reward):
        """Call once per episode with the episode's total reward. Returns the
        updated xi."""
        self.recent.append(float(episode_reward))
        avg = float(np.mean(self.recent))

        # Maintain running min/max of the smoothed return
        self.r_low = avg if self.r_low is None else min(self.r_low, avg)
        self.r_high = avg if self.r_high is None else max(self.r_high, avg)

        spread = self.r_high - self.r_low
        if spread > 1e-8:
            progress = (avg - self.r_low) / spread        # in [0, 1]
            self.xi = self.xi_max * float(np.clip(progress, 0.0, 1.0))
        else:
            self.xi = 0.0
        return self.xi
