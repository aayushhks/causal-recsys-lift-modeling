import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.ab_testing.bayesian_engine import BayesianABTester
import numpy as np


def run_simulation():
    tester = BayesianABTester()

    # 1. Simulate Data
    # Control: 10,000 views, 1,000 clicks (10% CTR)
    # Treatment: 10,000 views, 1,150 clicks (11.5% CTR) -> This IS better
    print(" Simulating A/B Test Data...")
    tester.update('control', impressions=10000, clicks=1000)
    tester.update('treatment', impressions=10000, clicks=1150)

    # 2. Evaluate
    results = tester.evaluate_experiment()

    print("\n--- Results ---")
    print(f"Prob Treatment > Control: {results['prob_being_best']:.4f}")
    print(f"Expected Lift:            {results['expected_lift']:.2%}")
    print(
        f"95% Interval:             [{results['lift_95_cred_interval'][0]:.2%}, {results['lift_95_cred_interval'][1]:.2%}]")

    if results['prob_being_best'] > 0.95:
        print("\n RESULT: Significant Win for Treatment!")
    else:
        print("\n RESULT: Not significant yet.")


if __name__ == "__main__":
    run_simulation()