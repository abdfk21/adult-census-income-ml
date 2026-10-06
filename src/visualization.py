"""
Visualization module for Adult Census Income Prediction.
Provides interactive Plotly visualizations for model metrics, confusion matrix,
ROC curves, probability gauges, feature importances, and exploratory data analysis.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Custom clean design palette
PALETTE = {
    'primary': '#4361EE',
    'secondary': '#3F37C9',
    'accent': '#4CC9F0',
    'success': '#10B981',
    'danger': '#EF4444',
    'warning': '#F59E0B',
    'dark': '#1E293B',
    'light': '#F8FAFC',
    'card_bg': '#FFFFFF',
    'border': '#E2E8F0'
}

CHART_THEME = "plotly_white"


def plot_metric_comparison_bars(results_df: pd.DataFrame, metric: str = 'F1') -> go.Figure:
    """
    Bar chart comparing a single metric across all models, highlighting the top model.
    """
    df_sorted = results_df.copy()
    if 'Model' not in df_sorted.columns:
        df_sorted['Model'] = df_sorted.index
        
    df_sorted = df_sorted.sort_values(metric, ascending=True)
    best_idx = df_sorted[metric].idxmax()
    
    colors = [
        PALETTE['success'] if idx == best_idx else '#94A3B8'
        for idx in df_sorted.index
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted[metric],
        y=df_sorted['Model'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='#CBD5E1', width=1)),
        text=[f"{v:.4f}" for v in df_sorted[metric]],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>" + f"{metric}: " + "%{x:.4f}<extra></extra>"
    ))
    
    max_val = df_sorted[metric].max()
    fig.update_layout(
        template=CHART_THEME,
        title=dict(
            text=f"<b>Model Comparison: {metric} Score</b> (Best Highlighted in Emerald)",
            font=dict(size=16, color=PALETTE['dark'])
        ),
        xaxis=dict(
            title=f"{metric} Score",
            range=[0, min(1.0, max_val * 1.15)],
            gridcolor='#F1F5F9'
        ),
        yaxis=dict(title=""),
        margin=dict(l=20, r=40, t=50, b=30),
        height=450
    )
    return fig


def plot_multi_metric_comparison(results_df: pd.DataFrame) -> go.Figure:
    """
    Multi-metric grouped bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    df = results_df.copy()
    if 'Model' not in df.columns:
        df['Model'] = df.index
        
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    color_map = {
        'Accuracy': '#4361EE',
        'Precision': '#06D6A0',
        'Recall': '#FFD166',
        'F1': '#EF476F',
        'ROC-AUC': '#118AB2'
    }
    
    fig = go.Figure()
    for m in metrics:
        if m in df.columns:
            fig.add_trace(go.Bar(
                name=m,
                x=df['Model'],
                y=df[m],
                marker_color=color_map.get(m, '#888'),
                hovertemplate=f"<b>%{{x}}</b><br>{m}: %{{y:.4f}}<extra></extra>"
            ))
            
    fig.update_layout(
        template=CHART_THEME,
        barmode='group',
        title=dict(
            text="<b>Side-by-Side Model Performance Across Classification Metrics</b>",
            font=dict(size=16, color=PALETTE['dark'])
        ),
        xaxis=dict(tickangle=-25, gridcolor='#F1F5F9'),
        yaxis=dict(title="Score", range=[0.5, 1.0], gridcolor='#F1F5F9'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=70, b=50),
        height=500
    )
    return fig


def plot_model_radar_comparison(results_df: pd.DataFrame, selected_models: List[str]) -> go.Figure:
    """
    Radar chart comparing selected models across multi-dimensional metrics.
    """
    categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    df = results_df.copy()
    if 'Model' not in df.columns:
        df['Model'] = df.index
        
    fig = go.Figure()
    colors = ['#4361EE', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
    
    for i, model_name in enumerate(selected_models):
        if model_name in df['Model'].values:
            row = df[df['Model'] == model_name].iloc[0]
            vals = [row[cat] for cat in categories]
            vals.append(vals[0])  # Close the radar loop
            
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill='toself',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2),
                opacity=0.65
            ))
            
    fig.update_layout(
        template=CHART_THEME,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.5, 1.0],
                gridcolor='#E2E8F0'
            )
        ),
        title=dict(text="<b>Radar Trade-off Evaluation</b>", font=dict(size=15)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=50, b=60),
        height=450
    )
    return fig


def plot_confusion_matrix_interactive(cm: np.ndarray, model_name: str = "Stacking Classifier") -> go.Figure:
    """
    Plots an annotated confusion matrix heatmap.
    """
    labels = ["<=50K (Negative)", ">50K (Positive)"]
    z_text = [
        [f"TN = {cm[0, 0]}<br>(True <=50K)", f"FP = {cm[0, 1]}<br>(False >50K)"],
        [f"FN = {cm[1, 0]}<br>(False <=50K)", f"TP = {cm[1, 1]}<br>(True >50K)"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=z_text,
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        colorscale='Viridis',
        showscale=True,
        hoverongaps=False
    ))
    
    fig.update_layout(
        template=CHART_THEME,
        title=dict(
            text=f"<b>Confusion Matrix: {model_name} (Test Split N={cm.sum()})</b>",
            font=dict(size=16)
        ),
        xaxis=dict(title="<b>Predicted Class</b>"),
        yaxis=dict(title="<b>True Ground Truth Class</b>", autorange="reversed"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=450
    )
    return fig


def plot_roc_curve_interactive(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    model_name: str = "Stacking Classifier"
) -> go.Figure:
    """
    Plots the ROC Curve against the random guessing baseline.
    """
    fig = go.Figure()
    
    # Baseline
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier (AUC = 0.500)',
        line=dict(color='#94A3B8', dash='dash', width=2)
    ))
    
    # Model ROC
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc_score:.3f})',
        line=dict(color=PALETTE['primary'], width=3),
        fill='tonexty',
        fillcolor='rgba(67, 97, 238, 0.1)'
    ))
    
    fig.update_layout(
        template=CHART_THEME,
        title=dict(text=f"<b>ROC Curve - {model_name}</b>", font=dict(size=16)),
        xaxis=dict(title="False Positive Rate (FPR)", range=[0, 1], gridcolor='#F1F5F9'),
        yaxis=dict(title="True Positive Rate (Sensitivity/Recall)", range=[0, 1.02], gridcolor='#F1F5F9'),
        legend=dict(x=0.55, y=0.15, bgcolor='rgba(255,255,255,0.85)'),
        margin=dict(l=30, r=30, t=50, b=40),
        height=450
    )
    return fig


def plot_probability_gauge(prob_over_50k: float) -> go.Figure:
    """
    Interactive indicator gauge for >$50K income likelihood.
    """
    pct = prob_over_50k * 100
    color = PALETTE['success'] if pct >= 50 else PALETTE['primary']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(size=42, color=PALETTE['dark'])),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#94A3B8"),
            bar=dict(color=color, thickness=0.3),
            bgcolor="white",
            borderwidth=1,
            bordercolor="#E2E8F0",
            steps=[
                dict(range=[0, 30], color="#F1F5F9"),
                dict(range=[30, 70], color="#E2E8F0"),
                dict(range=[70, 100], color="#FEF3C7")
            ],
            threshold=dict(
                line=dict(color="#EF4444", width=3),
                thickness=0.8,
                value=50.0
            )
        )
    ))
    
    fig.update_layout(
        template=CHART_THEME,
        title=dict(text="<b>Probability of Annual Income > $50K</b>", font=dict(size=15)),
        margin=dict(l=25, r=25, t=50, b=20),
        height=280
    )
    return fig


def plot_base_learner_breakdown(
    base_probs: Dict[str, float],
    final_prob: float
) -> go.Figure:
    """
    Horizontal bar chart decomposing ensemble probabilities from individual base learners.
    """
    names_map = {
        'lr': 'Logistic Regression',
        'rf': 'Tuned Random Forest',
        'xgb': 'Tuned XGBoost'
    }
    
    models = [names_map.get(k, k) for k in base_probs.keys()] + ['Stacking Ensemble (Meta)']
    probs = [base_probs[k] for k in base_probs.keys()] + [final_prob]
    
    colors = [
        '#64748B' for _ in range(len(base_probs))
    ] + [PALETTE['success'] if final_prob >= 0.5 else PALETTE['primary']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[p * 100 for p in probs],
        y=models,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p * 100:.1f}%" for p in probs],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>P(>50K): %{x:.2f}%<extra></extra>"
    ))
    
    fig.update_layout(
        template=CHART_THEME,
        title=dict(text="<b>Base Learner Probability Decomposition</b>", font=dict(size=15)),
        xaxis=dict(title="Estimated Probability of > $50K (%)", range=[0, 115], gridcolor='#F1F5F9'),
        yaxis=dict(title=""),
        margin=dict(l=10, r=30, t=50, b=30),
        height=280
    )
    return fig


def plot_feature_importance_bars(df_imp: pd.DataFrame, title: str) -> go.Figure:
    """
    Horizontal bar chart for constituent tree feature importances.
    """
    df_sorted = df_imp.sort_values('importance', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted['importance'],
        y=df_sorted['feature'],
        orientation='h',
        marker=dict(
            color=df_sorted['importance'],
            colorscale='Blues',
            line=dict(color='#CBD5E1', width=1)
        ),
        text=[f"{v:.4f}" for v in df_sorted['importance']],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    
    max_val = df_sorted['importance'].max()
    fig.update_layout(
        template=CHART_THEME,
        title=dict(text=f"<b>{title}</b>", font=dict(size=15)),
        xaxis=dict(title="Relative Importance Weight", range=[0, max_val * 1.25], gridcolor='#F1F5F9'),
        yaxis=dict(title=""),
        margin=dict(l=10, r=40, t=45, b=30),
        height=480
    )
    return fig


def plot_eda_class_donut(df: pd.DataFrame) -> go.Figure:
    """
    Donut chart of income class balance.
    """
    counts = df['income'].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=['#4361EE', '#10B981']),
        textinfo='label+percent+value',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        template=CHART_THEME,
        title=dict(text="<b>Income Class Distribution (Target Imbalance)</b>", font=dict(size=15)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig


def plot_eda_bivariate_histogram(
    df: pd.DataFrame,
    feature: str,
    title: Optional[str] = None
) -> go.Figure:
    """
    Histogram / distribution plot of numerical feature stratified by income class.
    """
    fig = px.histogram(
        df,
        x=feature,
        color="income",
        barmode="overlay",
        opacity=0.7,
        nbins=40,
        color_discrete_map={"<=50K": "#4361EE", ">50K": "#10B981"},
        title=title or f"<b>Distribution of {feature.capitalize()} by Income Level</b>",
        template=CHART_THEME
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=30),
        xaxis=dict(gridcolor='#F1F5F9'),
        yaxis=dict(title="Count", gridcolor='#F1F5F9'),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_eda_categorical_proportion(df: pd.DataFrame, cat_col: str) -> go.Figure:
    """
    100% stacked bar chart showing proportion of >50K across categories.
    """
    ct = pd.crosstab(df[cat_col], df['income'], normalize='index') * 100
    ct = ct.sort_values(by='>50K', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ct.index,
        x=ct['<=50K'],
        name='<=50K',
        orientation='h',
        marker_color='#4361EE'
    ))
    fig.add_trace(go.Bar(
        y=ct.index,
        x=ct['>50K'],
        name='>50K',
        orientation='h',
        marker_color='#10B981'
    ))
    
    fig.update_layout(
        template=CHART_THEME,
        barmode='stack',
        title=dict(text=f"<b>Proportion of Income Levels by {cat_col.replace('.', ' ').capitalize()}</b>", font=dict(size=15)),
        xaxis=dict(title="Percentage (%)", range=[0, 100], gridcolor='#F1F5F9'),
        yaxis=dict(title=""),
        margin=dict(l=10, r=20, t=50, b=30),
        height=max(360, len(ct) * 28),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
