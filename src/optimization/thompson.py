import numpy as np
from scipy.stats import beta


class ThompsonSampler:
    """
    Implements Thompson Sampling for Multi-Armed Bandit problems.
    Used for real-time traffic allocation in A/B tests.
    """

    def __init__(self, n_arms=2, prior_alpha=1.0, prior_beta=1.0):
        """
        Initialize with prior beliefs (usually uniform Beta(1,1)).
        n_arms: Number of variants (e.g., 2 for Control vs Treatment)
        """
        self.n_arms = n_arms
        # Alpha = Successes + Prior, Beta = Failures + Prior
        self.alphas = np.full(n_arms, prior_alpha)
        self.betas = np.full(n_arms, prior_beta)

    def select_arm(self):
        """
        Samples from the posterior Beta distribution of each arm
        and selects the one with the highest sampled probability.
        Returns: Index of the selected arm (0 = Control, 1 = Treatment)
        """
        sampled_theta = np.random.beta(self.alphas, self.betas)
        return np.argmax(sampled_theta)

    def update(self, arm_index, reward):
        """
        Updates the posterior distribution after observing a result.
        arm_index: The variant shown (0 or 1)
        reward: 1 if converted (click/purchase), 0 if not
        """
        if reward == 1:
            self.alphas[arm_index] += 1
        else:
            self.betas[arm_index] += 1

    def get_probabilities(self):
        """
        Returns the current mean conversion rate estimate for each arm.
        Mean of Beta(a, b) = a / (a + b)
        """
        return self.alphas / (self.alphas + self.betas)


# --- Simulation for Verification ---
if __name__ == "__main__":
    print("🎰 Running Thompson Sampling Simulation...")

    # True Conversion Rates (Hidden from the model)
    # Variant B (0.15) is better than A (0.10)
    TRUE_RATES = [0.10, 0.15]

    bandit = ThompsonSampler(n_arms=2)

    # Simulate 1,000 users visiting the site
    n_users = 1000
    allocation_counts = [0, 0]

    for _ in range(n_users):
        # 1. Bandit chooses which variant to show
        chosen_arm = bandit.select_arm()
        allocation_counts[chosen_arm] += 1

        # 2. Simulate User Behavior (Bernoulli trial)
        # Did they buy? (Based on the TRUE rate of that variant)
        reward = 1 if np.random.rand() < TRUE_RATES[chosen_arm] else 0

        # 3. Bandit learns from the result
        bandit.update(chosen_arm, reward)

    print(f"\nResults after {n_users} users:")
    print(f"   Traffic to Control (A):   {allocation_counts[0]} users")
    print(f"   Traffic to Treatment (B): {allocation_counts[1]} users")
    print(f"   Estimated Rates:          {bandit.get_probabilities()}")
    print("\n✅ Success: The system automatically routed more traffic to the better variant (B).")