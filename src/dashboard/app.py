# src/dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# 1. Add project root to path so we can import from src
# We are in src/dashboard/app.py, so we go up TWO levels to reach root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 2. FIXED IMPORTS (Matching the new folder structure)
from src.ab_testing.bayesian_engine import BayesianABTester
from src.causal.inference_engine import CausalInferenceEngine

# Configuration
st.set_page_config(page_title="Causal RecSys Engine", layout="wide")
st.title(" Causal Recommendation Engine: Lift & A/B Analysis")

# Tabs
tab1, tab2 = st.tabs([" A/B Test Live Monitor", " Causal Impact Analysis"])

# Tab 1: Bayesian A/B Test
with tab1:
    st.header("Real-time Bayesian A/B Test")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Control Group (A)")
        ctrl_imps = st.number_input("Impressions (A)", value=10000, step=100)
        ctrl_clicks = st.number_input("Clicks (A)", value=850, step=10)  # 8.5% CTR
    with col2:
        st.subheader("Treatment Group (B)")
        treat_imps = st.number_input("Impressions (B)", value=10000, step=100)
        treat_clicks = st.number_input("Clicks (B)", value=950, step=10)  # 9.5% CTR

    if st.button("Run Bayesian Analysis"):
        tester = BayesianABTester()
        tester.update('Control', ctrl_imps, ctrl_clicks)
        tester.update('Treatment', treat_imps, treat_clicks)

        res = tester.evaluate_experiment('Control', 'Treatment')

        # Metrics Display
        m1, m2, m3 = st.columns(3)
        m1.metric("Prob(B is Best)", f"{res['prob_being_best']:.2%}")
        m2.metric("Expected Lift", f"{res['expected_lift']:.2%}")
        m3.metric("95% Interval", f"[{res['lift_95_cred_interval'][0]:.2%}, {res['lift_95_cred_interval'][1]:.2%}]")

        # Visualization
        samples_A = tester.sample_posterior('Control')
        samples_B = tester.sample_posterior('Treatment')

        df_vis = pd.DataFrame({
            'CTR': list(samples_A) + list(samples_B),
            'Variant': ['Control'] * len(samples_A) + ['Treatment'] * len(samples_B)
        })

        fig = px.histogram(df_vis, x="CTR", color="Variant", barmode="overlay", nbins=100,
                           title="Posterior Distributions of CTR", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Causal Inference
with tab2:
    st.header("Propensity Score Stratification (DoWhy)")

    uploaded_file = st.file_uploader("Upload Experiment Data (CSV)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        treatment_col = st.selectbox("Treatment Column", df.columns)
        outcome_col = st.selectbox("Outcome Column", df.columns)
        confounders = st.multiselect("Confounders (Common Causes)", df.columns)

        if st.button("Estimate Causal Effect"):
            with st.spinner("Building Causal Graph..."):
                engine = CausalInferenceEngine(df)
                engine.create_model(treatment_col, outcome_col, confounders)
                engine.identify_effect()
                ate = engine.estimate_effect()

                st.success(f"✅ Estimated Average Treatment Effect (ATE): {ate:.4f}")

                st.info("Running Placebo Refutation...")
                refute = engine.refute_estimate()
                st.text(str(refute))