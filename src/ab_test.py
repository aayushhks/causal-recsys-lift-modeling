import numpy as np
from scipy.stats import beta


class BayesianABTester:
    def __init__(self, alpha_prior=1, beta_prior=1):
        """
        Initializes the tester with a Beta(alpha, beta) prior.
        Default is Beta(1,1) which is a uniform distribution (we know nothing).
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        # Storage for experiment groups
        self.groups = {}

    def add_variant(self, name):
        """Adds a new variant (e.g., 'control', 'treatment') to track."""
        self.groups[name] = {
            'impressions': 0,
            'clicks': 0,
            'alpha': self.alpha_prior,
            'beta': self.beta_prior
        }

    def update(self, variant, impressions, clicks):
        """
        Updates the posterior distribution for a variant with new data.
        """
        if variant not in self.groups:
            self.add_variant(variant)

        # Update counts
        self.groups[variant]['impressions'] += impressions
        self.groups[variant]['clicks'] += clicks

        # Update Beta parameters (Posterior = Prior + Likelihood)
        # Alpha captures successes (clicks)
        self.groups[variant]['alpha'] += clicks
        # Beta captures failures (no-clicks)
        self.groups[variant]['beta'] += (impressions - clicks)

    def sample_posterior(self, variant, n_samples=10000):
        """Draws random samples from the posterior Beta distribution."""
        g = self.groups[variant]
        return beta.rvs(g['alpha'], g['beta'], size=n_samples)

    def evaluate_experiment(self, control_name='control', treatment_name='treatment'):
        """
        Calculates the probability that Treatment beats Control.
        """
        print(f" Evaluating: {treatment_name} vs {control_name}")

        # Monte Carlo Simulation
        control_samples = self.sample_posterior(control_name)
        treatment_samples = self.sample_posterior(treatment_name)

        # Probability Treatment > Control
        prob_superior = (treatment_samples > control_samples).mean()

        # Expected Uplift (Relative)
        uplift = (treatment_samples - control_samples) / control_samples
        expected_uplift = uplift.mean()

        return {
            'prob_being_best': prob_superior,
            'expected_lift': expected_uplift,
            'lift_95_cred_interval': (np.percentile(uplift, 2.5), np.percentile(uplift, 97.5))
        }