import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CausalInferenceEngine:
    def __init__(self, data):
        """
        Expects a pandas DataFrame containing:
        - treatment (binary: 0/1)
        - outcome (binary: 0/1 or continuous)
        - common_causes (list of confounders)
        """
        self.data = data
        self.model = None
        self.identified_estimand = None
        self.estimate = None

    def create_model(self, treatment_col, outcome_col, common_causes):
        print(f"  Building Causal Graph: {treatment_col} -> {outcome_col}")

        # Define the Causal Graph
        # We assume common_causes affect BOTH treatment and outcome
        self.model = CausalModel(
            data=self.data,
            treatment=treatment_col,
            outcome=outcome_col,
            common_causes=common_causes
        )

        # Visualize the graph (Optional, requires pydot/graphviz)
        # self.model.view_model()

    def identify_effect(self):
        print(" Identifying Causal Effect...")
        self.identified_estimand = self.model.identify_effect(proceed_when_unidentifiable=True)
        # print(self.identified_estimand) # Debugging

    def estimate_effect(self, method="backdoor.propensity_score_stratification"):
        """
        Calculates the Average Treatment Effect (ATE).
        Methods:
        - 'backdoor.propensity_score_matching'
        - 'backdoor.propensity_score_stratification' (Robust default)
        - 'backdoor.linear_regression'
        """
        print(f" Estimating effect using {method}...")
        self.estimate = self.model.estimate_effect(
            self.identified_estimand,
            method_name=method
        )

        print(f"\n--- Causal Estimate ---")
        print(f"ATE (Average Treatment Effect): {self.estimate.value:.5f}")
        return self.estimate.value

    def refute_estimate(self):
        """
        Sanity Check: If we replace the treatment with a random placebo,
        the effect should go to zero. If it doesn't, our model is wrong.
        """
        print("  Running Placebo Refutation Test...")
        refute = self.model.refute_estimate(
            self.identified_estimand,
            self.estimate,
            method_name="placebo_treatment_refuter"
        )
        print(refute)
        return refute


# Usage Example
if __name__ == "__main__":
    np.random.seed(42)  # Reproducibility
    N = 5000
    df = pd.DataFrame({
        'is_loyal_customer': np.random.randint(0, 2, N),
    })

    # 1. Assign Treatment (biased by loyalty)
    # Loyal customers are 80% likely to get the ad (treatment=1)
    df['treatment'] = np.where(df['is_loyal_customer'] == 1,
                               np.random.choice([0, 1], N, p=[0.2, 0.8]),
                               np.random.choice([0, 1], N, p=[0.8, 0.2]))

    # 2. Assign Outcome (Probability of Conversion)
    # Baseline: 10%
    # Loyal Lift: +20%
    # Treatment Lift: +10% (This is what we want to find!)
    df['conversion_prob'] = 0.10 + (0.20 * df['is_loyal_customer']) + (0.10 * df['treatment'])

    # 3. Simulate Binary Outcome (Coin flip based on probability)
    df['conversion'] = np.random.binomial(1, df['conversion_prob'])

    # Run the Engine
    print(f"Dataset stats:\n{df.groupby('treatment')['conversion'].mean()}")

    engine = CausalInferenceEngine(df)
    engine.create_model(treatment_col='treatment', outcome_col='conversion', common_causes=['is_loyal_customer'])
    engine.identify_effect()
    engine.estimate_effect()
    engine.refute_estimate()