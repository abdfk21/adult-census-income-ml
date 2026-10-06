"""
Adult Census Income Predictor - Streamlit Application
A modern, dynamic Machine Learning dashboard and inference interface.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import (
    load_dataset,
    get_feature_metadata,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    EDUCATION_TO_NUM,
    CATEGORICAL_OPTIONS,
    FEATURE_DESCRIPTIONS
)
from src.prediction import IncomePredictionService
from src.visualization import (
    plot_metric_comparison_bars,
    plot_multi_metric_comparison,
    plot_model_radar_comparison,
    plot_confusion_matrix_interactive,
    plot_roc_curve_interactive,
    plot_probability_gauge,
    plot_base_learner_breakdown,
    plot_feature_importance_bars,
    plot_eda_class_donut,
    plot_eda_bivariate_histogram,
    plot_eda_categorical_proportion
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Adult Census Income Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern UI Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Card */
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 14px;
        padding: 24px 30px;
        color: #F8FAFC;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #334155;
    }
    .hero-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 6px;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 15px;
        color: #94A3B8;
        margin-bottom: 0px;
        line-height: 1.5;
    }
    
    /* Metric Card */
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        transition: transform 0.15s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 12px;
        color: #10B981;
        font-weight: 500;
        margin-top: 2px;
    }

    /* Result Cards */
    .result-card-pos {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-card-neg {
        background: linear-gradient(135deg, #F1F5F9 0%, #E2E8F0 100%);
        border: 2px solid #64748B;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-title {
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .result-value-pos {
        font-size: 34px;
        font-weight: 800;
        color: #047857;
        margin: 6px 0;
    }
    .result-value-neg {
        font-size: 34px;
        font-weight: 800;
        color: #1E293B;
        margin: 6px 0;
    }

    /* Explainer Box */
    .explainer-box {
        background: #F8FAFC;
        border-radius: 10px;
        border-left: 4px solid #4361EE;
        padding: 14px 18px;
        font-size: 13px;
        color: #334155;
        line-height: 1.6;
        margin-top: 10px;
    }
    
    /* Preset Profile Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 18px;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning model...")
def get_prediction_service() -> IncomePredictionService:
    return IncomePredictionService()


@st.cache_data(show_spinner="Loading model evaluation results...")
def get_model_results() -> pd.DataFrame:
    path = "results/model_results.csv"
    if not os.path.exists(path):
        path = "Data/adult_income_model_results.csv"
    df = pd.read_csv(path, index_col=0)
    return df


@st.cache_data(show_spinner="Loading test evaluation artifacts...")
def get_test_artifacts() -> dict:
    path = "results/test_evaluation_artifacts.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner="Loading dataset...")
def get_clean_data() -> pd.DataFrame:
    path = "Data/adult.csv"
    if not os.path.exists(path):
        path = "data/adult.csv"
    return load_dataset(path)


# ---------------------------------------------------------
# Load Resources
# ---------------------------------------------------------
try:
    service = get_prediction_service()
    results_df = get_model_results()
    test_artifacts = get_test_artifacts()
    metadata = get_feature_metadata()
except Exception as e:
    st.error(f"Failed to initialize core ML services: {e}")
    st.stop()


# ---------------------------------------------------------
# Preset Demographic Profiles for Rapid Testing
# ---------------------------------------------------------
PRESETS = {
    "Custom Input (Select below)": None,
    "Executive / Tech Lead (High Earner Pattern)": {
        "age": 45,
        "workclass": "Private",
        "fnlwgt": 178356,
        "education": "Masters",
        "education.num": 14,
        "marital.status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital.gain": 15024,
        "capital.loss": 0,
        "hours.per.week": 50,
        "native.country": "United-States"
    },
    "Entry-Level Retail Associate (Early Career Pattern)": {
        "age": 22,
        "workclass": "Private",
        "fnlwgt": 194820,
        "education": "HS-grad",
        "education.num": 9,
        "marital.status": "Never-married",
        "occupation": "Sales",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital.gain": 0,
        "capital.loss": 0,
        "hours.per.week": 28,
        "native.country": "United-States"
    },
    "Senior Academic / Professor (High Education)": {
        "age": 52,
        "workclass": "State-gov",
        "fnlwgt": 182340,
        "education": "Doctorate",
        "education.num": 16,
        "marital.status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital.gain": 7688,
        "capital.loss": 0,
        "hours.per.week": 48,
        "native.country": "United-States"
    },
    "Self-Employed Craftsman (Moderate Hours, Capital Loss)": {
        "age": 39,
        "workclass": "Self-emp-not-inc",
        "fnlwgt": 210450,
        "education": "Some-college",
        "education.num": 10,
        "marital.status": "Divorced",
        "occupation": "Craft-repair",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital.gain": 0,
        "capital.loss": 1887,
        "hours.per.week": 42,
        "native.country": "United-States"
    }
}


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/graph.png", width=64)
    st.title("Adult Census ML")
    st.caption("Binary Income Classification System")
    
    st.markdown("---")
    
    # Navigation
    app_mode = st.radio(
        "Navigation",
        [
            "💼 Income Predictor",
            "📊 Batch Predictions",
            "🏆 Model Benchmarking",
            "🔍 Dataset Explorer",
            "📐 System Architecture"
        ],
        index=0
    )
    
    st.markdown("---")
    
    # Model Badge Information
    st.markdown("### Production Model")
    st.info(
        "**Stacking Ensemble**\n\n"
        "• Base: LogReg + Tuned RF + Tuned XGB\n\n"
        "• Meta-Learner: Logistic Regression\n\n"
        "• F1 Score: **0.7340** | Acc: **87.59%**\n\n"
        "• ROC-AUC: **0.9302**"
    )
    
    st.markdown("---")
    st.caption("UCI Adult Census Income Dataset | N=30,139 records")


# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Adult Census Income Predictor</div>
    <div class="hero-subtitle">
        High-precision classical machine learning system predicting annual earnings classification 
        (<b>&gt; $50K</b> vs. <b>&le; $50K</b>) based on demographic, occupational, and capital attributes.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# PAGE 1: INCOME PREDICTOR
# =========================================================
if app_mode == "💼 Income Predictor":
    st.markdown("### 🎯 Single-Person Income Evaluation")
    st.write(
        "Enter individual demographic, educational, and financial attributes or pick a quick profile preset "
        "to evaluate the model's income prediction and base-learner decomposition."
    )
    
    # Quick Preset Selector
    c_preset, _ = st.columns([2, 1])
    with c_preset:
        selected_preset_key = st.selectbox(
            "⚡ Quick Preset Scenarios:",
            list(PRESETS.keys()),
            index=0
        )
    
    preset_vals = PRESETS[selected_preset_key]
    
    # Form Layout
    with st.form("prediction_form"):
        st.markdown("#### 1. Demographic & Household Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age_val = preset_vals["age"] if preset_vals else 38
            age = st.number_input("Age (years)", min_value=17, max_value=90, value=age_val, step=1)
            
        with col2:
            sex_val = preset_vals["sex"] if preset_vals else "Male"
            sex = st.selectbox("Biological Sex", metadata["categorical_options"]["sex"], index=metadata["categorical_options"]["sex"].index(sex_val))
            
        with col3:
            race_val = preset_vals["race"] if preset_vals else "White"
            race = st.selectbox("Race Demographic", metadata["categorical_options"]["race"], index=metadata["categorical_options"]["race"].index(race_val))
            
        with col4:
            country_val = preset_vals["native.country"] if preset_vals else "United-States"
            country_idx = metadata["categorical_options"]["native.country"].index(country_val) if country_val in metadata["categorical_options"]["native.country"] else 0
            native_country = st.selectbox("Native Country", metadata["categorical_options"]["native.country"], index=country_idx)

        col5, col6, col7 = st.columns(3)
        with col5:
            marital_val = preset_vals["marital.status"] if preset_vals else "Married-civ-spouse"
            marital_status = st.selectbox("Marital Status", metadata["categorical_options"]["marital.status"], index=metadata["categorical_options"]["marital.status"].index(marital_val))
            
        with col6:
            rel_val = preset_vals["relationship"] if preset_vals else "Husband"
            relationship = st.selectbox("Household Relationship", metadata["categorical_options"]["relationship"], index=metadata["categorical_options"]["relationship"].index(rel_val))
            
        with col7:
            fnlwgt_val = preset_vals["fnlwgt"] if preset_vals else 189795
            fnlwgt = st.number_input("Census Sampling Weight (fnlwgt)", min_value=10000, max_value=1500000, value=fnlwgt_val, step=5000, help="Estimated population weight represented by this record in Census sampling.")

        st.markdown("#### 2. Education & Employment Attributes")
        col8, col9, col10, col11 = st.columns(4)
        
        with col8:
            edu_val = preset_vals["education"] if preset_vals else "Bachelors"
            education = st.selectbox("Education Level", metadata["categorical_options"]["education"], index=metadata["categorical_options"]["education"].index(edu_val))
            
        with col9:
            # Auto-sync education number with selected education
            default_edu_num = EDUCATION_TO_NUM.get(education, 10)
            edu_num_val = preset_vals["education.num"] if preset_vals else default_edu_num
            education_num = st.number_input("Education Num (Years / Level)", min_value=1, max_value=16, value=edu_num_val, step=1, help="Numeric ordinal coding corresponding to formal education.")
            
        with col10:
            workclass_val = preset_vals["workclass"] if preset_vals else "Private"
            workclass = st.selectbox("Employment Sector (Workclass)", metadata["categorical_options"]["workclass"], index=metadata["categorical_options"]["workclass"].index(workclass_val))
            
        with col11:
            occ_val = preset_vals["occupation"] if preset_vals else "Exec-managerial"
            occupation = st.selectbox("Occupation Category", metadata["categorical_options"]["occupation"], index=metadata["categorical_options"]["occupation"].index(occ_val))

        st.markdown("#### 3. Capital & Working Schedule")
        col12, col13, col14 = st.columns(3)
        
        with col12:
            gain_val = preset_vals["capital.gain"] if preset_vals else 0
            capital_gain = st.number_input("Capital Gains (USD / Year)", min_value=0, max_value=100000, value=gain_val, step=500, help="Profits from sale of investments, stocks, or real estate.")
            
        with col13:
            loss_val = preset_vals["capital.loss"] if preset_vals else 0
            capital_loss = st.number_input("Capital Losses (USD / Year)", min_value=0, max_value=5000, value=loss_val, step=100, help="Financial losses from investment sales.")
            
        with col14:
            hours_val = preset_vals["hours.per.week"] if preset_vals else 40
            hours_per_week = st.number_input("Weekly Work Hours", min_value=1, max_value=99, value=hours_val, step=1)

        submitted = st.form_submit_button("🚀 Run Stacking Model Prediction", use_container_width=True)

    if submitted:
        input_data = {
            "age": age,
            "workclass": workclass,
            "fnlwgt": fnlwgt,
            "education": education,
            "education.num": education_num,
            "marital.status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital.gain": capital_gain,
            "capital.loss": capital_loss,
            "hours.per.week": hours_per_week,
            "native.country": native_country
        }
        
        with st.spinner("Executing pipeline transformation and ensemble inference..."):
            pred_res = service.predict_single(input_data)
            
        st.markdown("---")
        st.markdown("### 📋 Prediction Outcome")
        
        r_col1, r_col2 = st.columns([1, 1])
        
        with r_col1:
            if pred_res["prediction"] == 1:
                st.markdown(f"""
                <div class="result-card-pos">
                    <div class="result-title" style="color: #065F46;">Classification Decision</div>
                    <div class="result-value-pos">&gt; $50,000 / year</div>
                    <div style="color: #047857; font-size: 15px; font-weight: 600;">
                        Model Confidence: {pred_res['confidence']*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-neg">
                    <div class="result-title" style="color: #334155;">Classification Decision</div>
                    <div class="result-value-neg">&le; $50,000 / year</div>
                    <div style="color: #475569; font-size: 15px; font-weight: 600;">
                        Model Confidence: {pred_res['confidence']*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("""
            <div class="explainer-box">
                <b>⚠️ Operational Guidance:</b> Predictions represent statistical likelihoods estimated by 
                an ensemble trained on the 1994 UCI Adult Census dataset. Outputs reflect demographic and financial correlations 
                and should not be treated as deterministic financial truth.
            </div>
            """, unsafe_allow_html=True)
            
        with r_col2:
            fig_gauge = plot_probability_gauge(pred_res["prob_over_50k"])
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🔬 Explainability: Why this prediction?")
        
        tab_decomp, tab_meta, tab_drivers = st.tabs([
            "📊 Base Learner Consensus",
            "⚙️ Stacking Meta-Learner",
            "🌲 Key Tree Feature Drivers"
        ])
        
        with tab_decomp:
            st.write(
                "The Stacking Classifier combines diverse learning algorithms. "
                "Below are the individual positive-class (>50K) probabilities estimated by each base model:"
            )
            fig_base = plot_base_learner_breakdown(
                pred_res["base_learner_probs"],
                pred_res["prob_over_50k"]
            )
            st.plotly_chart(fig_base, use_container_width=True)
            
            c_lr, c_rf, c_xgb = st.columns(3)
            with c_lr:
                lr_p = pred_res["base_learner_probs"].get("lr", 0.0)
                st.metric("Logistic Regression", f"{lr_p*100:.1f}%", help="Linear boundary on scaled numerical + one-hot features")
            with c_rf:
                rf_p = pred_res["base_learner_probs"].get("rf", 0.0)
                st.metric("Tuned Random Forest", f"{rf_p*100:.1f}%", help="Ensemble of 413 bagged decision trees")
            with c_xgb:
                xgb_p = pred_res["base_learner_probs"].get("xgb", 0.0)
                st.metric("Tuned XGBoost", f"{xgb_p*100:.1f}%", help="Gradient boosted trees with shrinkage 0.077")

        with tab_meta:
            st.write(
                "**Meta-Learner Blending Mechanics:**\n\n"
                "The final classification is produced by a Meta-Logistic Regression model trained with 3-fold cross-validation. "
                "Instead of simple average voting, it weights each base model's probability distribution:"
            )
            weights = pred_res.get("meta_weights", {})
            intercept = pred_res.get("meta_intercept", 0.0)
            
            w_df = pd.DataFrame([
                {"Base Model": "Logistic Regression (lr)", "Meta Coefficient": weights.get("lr", 0.2236), "Relative Influence": "Low (0.22)"},
                {"Base Model": "Tuned Random Forest (rf)", "Meta Coefficient": weights.get("rf", 2.0105), "Relative Influence": "Moderate-High (2.01)"},
                {"Base Model": "Tuned XGBoost (xgb)", "Meta Coefficient": weights.get("xgb", 4.3405), "Relative Influence": "Dominant (4.34)"}
            ])
            st.dataframe(w_df, use_container_width=True)
            st.caption(f"Meta-Learner Log-Odds Intercept: {intercept:.4f}. Tuned XGBoost carries the highest predictive weight in resolving edge cases.")

        with tab_drivers:
            st.write(
                "**Global Feature Importances from Constituent Tree Models:**\n\n"
                "While Stacking blends multiple model families, the constituent tree-based models provide transparent insight "
                "into which attributes drive non-linear splits:"
            )
            col_rf_imp, col_xgb_imp = st.columns(2)
            tree_importances = service.get_global_feature_importances(top_n=10)
            
            with col_rf_imp:
                if "Random Forest" in tree_importances:
                    fig_rf_imp = plot_feature_importance_bars(
                        tree_importances["Random Forest"],
                        "Tuned Random Forest Top 10 Features"
                    )
                    st.plotly_chart(fig_rf_imp, use_container_width=True)
                    
            with col_xgb_imp:
                if "XGBoost" in tree_importances:
                    fig_xgb_imp = plot_feature_importance_bars(
                        tree_importances["XGBoost"],
                        "Tuned XGBoost Top 10 Features"
                    )
                    st.plotly_chart(fig_xgb_imp, use_container_width=True)


# =========================================================
# PAGE 2: BATCH PREDICTIONS
# =========================================================
elif app_mode == "📊 Batch Predictions":
    st.markdown("### 📁 Batch Prediction & Evaluation Sandbox")
    st.write(
        "Upload a CSV containing demographic records to compute predictions, probabilities, "
        "and summary confidence metrics in bulk."
    )
    
    col_up, col_dl = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])
    
    with col_dl:
        st.write("Need sample data?")
        try:
            sample_data = get_clean_data().head(10).drop(columns=['income'], errors='ignore')
            sample_csv = sample_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download 10-Row Sample Template",
                data=sample_csv,
                file_name="adult_income_template.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception:
            pass
            
    if uploaded_file is not None:
        try:
            batch_raw = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_raw)} records successfully.")
            
            with st.spinner("Processing batch inference..."):
                batch_preds = service.predict_batch(batch_raw)
                
            # Summary KPIs
            k1, k2, k3, k4 = st.columns(4)
            pct_over_50k = (batch_preds['Predicted_Income_Class'] == 1).mean() * 100
            mean_conf = batch_preds['Model_Confidence'].mean() * 100
            
            k1.metric("Total Records Processed", f"{len(batch_preds):,}")
            k2.metric("Classified >$50K", f"{pct_over_50k:.1f}%")
            k3.metric("Classified <=$50K", f"{100-pct_over_50k:.1f}%")
            k4.metric("Mean Model Confidence", f"{mean_conf:.1f}%")
            
            st.markdown("#### Scored Dataset Preview")
            st.dataframe(batch_preds.head(50), use_container_width=True)
            
            # Export scored data
            scored_csv = batch_preds.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Full Scored Predictions CSV",
                data=scored_csv,
                file_name="scored_adult_income_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing batch CSV: {e}")


# =========================================================
# PAGE 3: MODEL BENCHMARKING
# =========================================================
elif app_mode == "🏆 Model Benchmarking":
    st.markdown("### 🏆 Comprehensive Model Benchmarks & Comparison")
    st.write(
        "Evaluation results across all 9 experimental configurations evaluated on the identical "
        "20% stratified test partition (N = 6,028)."
    )
    
    # Leaderboard Cards
    best_row = results_df.loc["Stacking"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Overall Model", "Stacking Classifier", "Selected")
    c2.metric("Top F1-Score", f"{best_row['F1']:.4f}", "+0.49% over XGB")
    c3.metric("Top Accuracy", f"{best_row['Accuracy']*100:.2f}%", "87.59%")
    c4.metric("ROC-AUC", f"{best_row['ROC-AUC']:.4f}", "0.9302")
    c5.metric("Top Test Recall", "80.41%", "Tuned RF")
    
    st.markdown("---")
    
    # Table View
    st.markdown("#### 1. Official Evaluation Leaderboard")
    
    # Format and highlight
    st.dataframe(
        results_df.style
        .format({
            "MSE": "{:.4f}",
            "RMSE": "{:.4f}",
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        })
        .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], color="#D1FAE5")
        .highlight_min(subset=["MSE", "RMSE"], color="#FEF3C7"),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Comparative Visualizations
    st.markdown("#### 2. Performance Comparison Visualizations")
    metric_choice = st.selectbox(
        "Select metric for individual ranking bar chart:",
        ["F1", "Accuracy", "ROC-AUC", "Precision", "Recall"],
        index=0
    )
    
    fig_single = plot_metric_comparison_bars(results_df, metric_choice)
    st.plotly_chart(fig_single, use_container_width=True)
    
    fig_multi = plot_multi_metric_comparison(results_df)
    st.plotly_chart(fig_multi, use_container_width=True)
    
    # Radar Chart
    st.markdown("#### 3. Multi-Dimensional Trade-off Radar")
    models_to_radar = st.multiselect(
        "Select models for radar overlay:",
        results_df.index.tolist(),
        default=["Stacking", "Tuned XGBoost", "Tuned Random Forest", "Logistic Regression"]
    )
    if models_to_radar:
        fig_radar = plot_model_radar_comparison(results_df, models_to_radar)
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    
    # Best Model In-Depth Diagnostics
    st.markdown("#### 4. Best Model Diagnostics (Stacking Classifier)")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        if "cm" in test_artifacts:
            cm_arr = np.array(test_artifacts["cm"])
            fig_cm = plot_confusion_matrix_interactive(cm_arr, "Stacking Classifier")
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.caption(
                f"**Confusion Matrix Breakdown (Test Set N={cm_arr.sum()}):**\n\n"
                f"• True Negatives (<=50K correct): **{cm_arr[0,0]}** (93.8% specificity)\n\n"
                f"• False Positives (<=50K classified as >50K): **{cm_arr[0,1]}**\n\n"
                f"• False Negatives (>50K missed): **{cm_arr[1,0]}**\n\n"
                f"• True Positives (>50K identified): **{cm_arr[1,1]}** (68.8% recall)"
            )
            
    with d_col2:
        if "fpr" in test_artifacts and "tpr" in test_artifacts:
            fig_roc = plot_roc_curve_interactive(
                np.array(test_artifacts["fpr"]),
                np.array(test_artifacts["tpr"]),
                test_artifacts["auc"],
                "Stacking Classifier"
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption(
                "**ROC-AUC Analysis:** The area under the ROC curve of 0.9302 indicates exceptional ranking capability "
                "across varying decision thresholds, far superior to random classification (0.50)."
            )


# =========================================================
# PAGE 4: DATASET EXPLORER & EDA
# =========================================================
elif app_mode == "🔍 Dataset Explorer":
    st.markdown("### 🔍 Exploratory Data Analysis & Feature Distributions")
    st.write(
        "Interactive exploration of the UCI Adult Census Income dataset (30,139 cleaned records post missing-value removal)."
    )
    
    try:
        clean_df = get_clean_data()
        
        # Top KPI stats
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Cleaned Records", f"{len(clean_df):,}")
        k2.metric("Original Raw Records", "32,561", "-2,422 dropped")
        k3.metric("<=50K (Majority Class)", f"{(clean_df['income']=='<=50K').sum():,} (75.1%)")
        k4.metric(">50K (Minority Class)", f"{(clean_df['income']=='>50K').sum():,} (24.9%)")
        
        st.markdown("---")
        
        # Interactive Filter Drawer
        with st.expander("🛠️ Interactive Data Filter Panel", expanded=False):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                age_range = st.slider("Age Filter Range:", 17, 90, (20, 65))
            with f_col2:
                sel_workclass = st.multiselect("Filter Workclass:", sorted(clean_df['workclass'].unique()), default=[])
            with f_col3:
                sel_income = st.multiselect("Filter Income Level:", ["<=50K", ">50K"], default=[])

            filtered_df = clean_df.copy()
            filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
            if sel_workclass:
                filtered_df = filtered_df[filtered_df['workclass'].isin(sel_workclass)]
            if sel_income:
                filtered_df = filtered_df[filtered_df['income'].isin(sel_income)]
                
            st.caption(f"Showing {len(filtered_df):,} matching rows.")
            st.dataframe(filtered_df.head(20), use_container_width=True)

        st.markdown("#### Visual Explorations")
        
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            fig_donut = plot_eda_class_donut(clean_df)
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with v_col2:
            fig_age = plot_eda_bivariate_histogram(clean_df, "age", "Age Distribution by Income Class")
            st.plotly_chart(fig_age, use_container_width=True)

        v_col3, v_col4 = st.columns(2)
        with v_col3:
            fig_edu = plot_eda_categorical_proportion(clean_df, "education")
            st.plotly_chart(fig_edu, use_container_width=True)
            
        with v_col4:
            fig_hours = plot_eda_bivariate_histogram(clean_df, "hours.per.week", "Work Hours Distribution by Income")
            st.plotly_chart(fig_hours, use_container_width=True)

        # Correlation Heatmap for Numerical Features
        st.markdown("#### Numerical Feature Correlation Matrix")
        corr_matrix = clean_df[NUMERICAL_COLS].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Heatmap (Numerical Features)"
        )
        fig_corr.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading EDA views: {e}")


# =========================================================
# PAGE 5: SYSTEM ARCHITECTURE
# =========================================================
elif app_mode == "📐 System Architecture":
    st.markdown("### 📐 ML Pipeline Architecture & Engineering Design")
    st.write(
        "Detailed technical specification of the data pipeline, preprocessing transformers, "
        "hyperparameter optimization, and ensemble stacking implementation."
    )
    
    st.markdown("#### 1. End-to-End Pipeline Workflow")
    st.markdown("""
    ```mermaid
    flowchart TD
        A["UCI Adult Census Raw Data (32,561 records)"] --> B["Missing Value Removal (? -> NA -> dropna)"]
        B --> C["Deduplication (drop_duplicates -> 30,139 records)"]
        C --> D["Stratified Train/Test Split (80/20, Stratified by Income)"]
        D --> E["Training Partition (24,111 samples)"]
        D --> F["Holdout Test Partition (6,028 samples)"]
        
        E --> G["Preprocessors & Encoders"]
        G --> H1["StandardScaler (Linear/Distance Models)"]
        G --> H2["One-Hot Dummy Matrix (96 Columns)"]
        
        H1 & H2 --> I1["Logistic Regression (Base 1)"]
        H1 & H2 --> I2["Tuned Random Forest (Base 2, 413 trees)"]
        H1 & H2 --> I3["Tuned XGBoost (Base 3, 408 trees)"]
        
        I1 & I2 & I3 --> J["Stacking Meta-Learner (LogisticRegression, cv=3)"]
        J --> K["Production Stacking Classifier (F1: 0.7340, AUC: 0.9302)"]
    ```
    """)
    
    st.markdown("---")
    st.markdown("#### 2. Key Hyperparameters Extracted from Optimization")
    
    col_rf_param, col_xgb_param = st.columns(2)
    with col_rf_param:
        st.info("""
        **Tuned Random Forest Best Parameters:**
        - `n_estimators`: 413
        - `max_depth`: 30
        - `max_features`: 'sqrt'
        - `min_samples_split`: 13
        - `min_samples_leaf`: 1
        - `class_weight`: 'balanced'
        - *Best CV F1*: **0.7126**
        """)
        
    with col_xgb_param:
        st.info("""
        **Tuned XGBoost Best Parameters:**
        - `n_estimators`: 408
        - `max_depth`: 5
        - `learning_rate`: 0.0767
        - `subsample`: 0.9880
        - `colsample_bytree`: 0.6400
        - `min_child_weight`: 6
        - `gamma`: 0.2296
        """)

    st.markdown("---")
    st.markdown("#### 3. Model Assumptions, Biases & Practical Limitations")
    st.warning("""
    **Responsible AI & Dataset Limitations:**
    1. **Historical Nature (1994 US Census):** Income thresholds ($50,000 in 1994 represents significantly higher purchasing power today).
    2. **Class Imbalance:** 75.1% of records earn <=$50K, meaning naive accuracy (75.1%) is misleading; F1 and ROC-AUC are paramount.
    3. **Missing-Value Removal:** Dropping records with `?` reduced data from 32,561 to 30,139 (7.4% reduction), primarily in `workclass`, `occupation`, and `native.country`.
    4. **Demographic Biases:** The data reflects historical socioeconomic disparities across gender, race, and marital roles. Models must not be used for discriminatory lending or automated hiring.
    """)
