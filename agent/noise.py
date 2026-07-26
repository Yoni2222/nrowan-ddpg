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

    def __init__(self, xi_max=1.0, window=20, inf_R=None, sup_R=None):
        """If inf_R and sup_R are given, normalize against these KNOWN fixed
        bounds -- the paper's original eq. 12. Otherwise fall back to online
        normalization against the running min/max of the smoothed return.

        The fixed-bounds form matters: with running min/max, the FIRST success
        ever makes the current average equal the running max, so xi jumps
        straight to xi_max and the D penalty crushes sigma to zero within a
        couple of episodes -- killing exploration exactly when the agent first
        finds the goal (observed on sparse MountainCar: one success at ep 91
        collapsed sigma from 0.265 to ~0). With known bounds (e.g. success
        rate in [0, 1]), one success in a 20-episode window yields
        xi = 0.05 * xi_max -- gentle annealing proportional to competence."""
        self.xi_max = xi_max          # upper bound on the noise-reduction weight
        self.window = window
        self.recent = deque(maxlen=window)
        self.inf_R = inf_R
        self.sup_R = sup_R
        self.r_low = None
        self.r_high = None
        self.xi = 0.0

    def update(self, episode_reward):
        """Call once per episode with the episode's total reward (or success
        indicator). Returns the updated xi."""
        self.recent.append(float(episode_reward))
        avg = float(np.mean(self.recent))

        if self.inf_R is not None and self.sup_R is not None:
            # Paper eq. 12: normalize against known fixed reward bounds
            spread = self.sup_R - self.inf_R
            progress = (avg - self.inf_R) / spread
            self.xi = self.xi_max * float(np.clip(progress, 0.0, 1.0))
            return self.xi

        # Fallback: online normalization against running min/max
        self.r_low = avg if self.r_low is None else min(self.r_low, avg)
        self.r_high = avg if self.r_high is None else max(self.r_high, avg)

        spread = self.r_high - self.r_low
        if spread > 1e-8:
            progress = (avg - self.r_low) / spread        # in [0, 1]
            self.xi = self.xi_max * float(np.clip(progress, 0.0, 1.0))
        else:
            self.xi = 0.0
        return self.xi
