import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_qini(y_true, uplift_score, treatment, plot=False):
    """
    Calculates Qini Coefficient and optionally plots the Qini Curve.

    Args:
        y_true: Binary outcome (did they convert?)
        uplift_score: Predicted lift (treatment effect)
        treatment: Binary treatment indicator (0=Control, 1=Treatment)
    """
    # Create a dataframe for sorting
    data = pd.DataFrame({
        'y': y_true,
        'uplift': uplift_score,
        't': treatment
    }).sort_values('uplift', ascending=False)

    # Cumulative stats
    data['n_t'] = data['t'].cumsum()  # Count of treated users
    data['n_c'] = (1 - data['t']).cumsum()  # Count of control users
    data['y_t'] = (data['y'] * data['t']).cumsum()  # Count of treated conversions
    data['y_c'] = (data['y'] * (1 - data['t'])).cumsum()  # Count of control conversions

    # Prevent division by zero
    data['n_t'] = data['n_t'].replace(0, 1)
    data['n_c'] = data['n_c'].replace(0, 1)

    # Calculate incremental gains (Qini Curve)
    # Gain = (Conversions/Treated) - (Conversions/Control) scaled by total population
    N_t_total = data['t'].sum()
    N_c_total = (1 - data['t']).sum()

    # Standardized Qini Curve calculation
    data['curve'] = data['y_t'] - (data['y_c'] * N_t_total / N_c_total)

    # Area Under Curve (Approximate)
    area = np.trapz(data['curve'], dx=1 / len(data))

    print(f"    Qini Score (AUUC): {area:.4f}")

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(data)), data['curve'], label='Model')
        plt.plot([0, len(data)], [0, data['curve'].iloc[-1]], 'r--', label='Random')
        plt.title(f"Qini Curve (Area={area:.3f})")
        plt.xlabel("Population Targeted (Highest Lift First)")
        plt.ylabel("Cumulative Incremental Gains")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("models/qini_curve.png")
        print("    Saved Qini plot to models/qini_curve.png")

    return area