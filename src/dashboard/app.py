import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.ab_testing.bayesian_engine import BayesianABTester
from src.causal.inference_engine import CausalInferenceEngine

st.set_page_config(page_title="Causal RecSys Engine", layout="wide")
st.title("Causal Recommendation Engine")

tab1, tab2 = st.tabs([" A/B Test Monitor", " Causal Inference"])

# Tab 1: A/B Test
with tab1:
    st.header("Bayesian A/B Test Simulation")
    col1, col2 = st.columns(2)
    with col1:
        ctrl_imps = st.number_input("Impressions (A)", 10000, step=100)
        ctrl_clicks = st.number_input("Clicks (A)", 850, step=10)
    with col2:
        treat_imps = st.number_input("Impressions (B)", 10000, step=100)
        treat_clicks = st.number_input("Clicks (B)", 950, step=10)

    if st.button("Run Bayesian Analysis"):
        tester = BayesianABTester()
        tester.update('Control', ctrl_imps, ctrl_clicks)
        tester.update('Treatment', treat_imps, treat_clicks)
        res = tester.evaluate_experiment('Control', 'Treatment')

        m1, m2, m3 = st.columns(3)
        m1.metric("Prob(B > A)", f"{res['prob_being_best']:.2%}")
        m2.metric("Expected Lift", f"{res['expected_lift']:.2%}")
        m3.metric("95% Interval", f"[{res['lift_95_cred_interval'][0]:.2%}, {res['lift_95_cred_interval'][1]:.2%}]")

        samples_A = tester.sample_posterior('Control')
        samples_B = tester.sample_posterior('Treatment')
        df_vis = pd.DataFrame({
            'CTR': list(samples_A) + list(samples_B),
            'Variant': ['Control'] * len(samples_A) + ['Treatment'] * len(samples_B)
        })
        fig = px.histogram(df_vis, x="CTR", color="Variant", barmode="overlay", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Causal Inference
with tab2:
    st.header("Causal Effect Estimation (DoWhy)")

    # UPDATED: Now accepts 'parquet' AND 'csv'
    uploaded_file = st.file_uploader("Upload Processed Data", type=["parquet", "csv"])

    if uploaded_file is not None:
        try:
            # Load based on extension
            if uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            st.write("Preview:", df.head(3))

            # Dropdown Defaults
            all_cols = df.columns.tolist()
            treat_default = all_cols.index('variant') if 'variant' in all_cols else 0
            outcome_default = all_cols.index('purchased') if 'purchased' in all_cols else 0

            col1, col2 = st.columns(2)
            with col1:
                treatment_col = st.selectbox("Treatment (Cause)", all_cols, index=treat_default)
            with col2:
                outcome_col = st.selectbox("Outcome (Effect)", all_cols, index=outcome_default)

            confounders = st.multiselect("Confounders", [c for c in all_cols if c not in [treatment_col, outcome_col]])

            if st.button("Estimate Causal Effect"):
                if treatment_col == outcome_col:
                    st.error(" Treatment and Outcome cannot be the same!")
                else:
                    # Convert 'Treatment'/'Control' strings to 1/0
                    if df[treatment_col].dtype == 'object':
                        df['treatment_binary'] = (df[treatment_col] == 'Treatment').astype(int)
                        actual_treatment = 'treatment_binary'
                    else:
                        actual_treatment = treatment_col

                    with st.spinner("Building Causal Graph..."):
                        engine = CausalInferenceEngine(df)
                        engine.create_model(actual_treatment, outcome_col, confounders)
                        engine.identify_effect()
                        ate = engine.estimate_effect()

                        st.success(f" Estimated Average Treatment Effect (ATE): {ate:.4f}")
                        st.info("Running Refutation Test...")
                        refute = engine.refute_estimate()
                        st.text(str(refute))

        except Exception as e:
            st.error(f"Error: {e}")
