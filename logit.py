import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go

# Set application layout and page config
st.set_page_config(page_title="Microeconometrics: Logistic Car Ownership Model", layout="wide")

# GLOBAL TYPOGRAPHY LAYER: Enforces Times New Roman globally across text fields
st.markdown(
    """
    <style>
    p, li, span, label, .stMarkdown p, .stVerticalBlock div p {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 18.5px !important;
    }
    .stMarkdown h1, h1 {
        font-family: 'Times New Roman', serif !important;
        font-size: 26px !important;
        font-weight: bold !important;
    }
    .stMarkdown h2, h2 {
        font-family: 'Times New Roman', serif !important;
        font-size: 22px !important;
        font-weight: bold !important;
    }
    .stMarkdown h3, h3 {
        font-family: 'Times New Roman', serif !important;
        font-size: 20px !important;
        font-weight: bold !important;
    }
    .stDataFrame table, .stDataFrame th, .stDataFrame td {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 14.5px !important;
    }
    pre, code {
        font-family: monospace !important;
        font-size: 15px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Microeconometrics Dashboard: Logistic Car Ownership Model")
st.subheader("An Interactive Evaluation of Binary Choice Frameworks for Grouped Data")
st.markdown("---")

# 1. Dataset array values matching the matrix layout exactly
obs = list(range(1, 11))
income = [6, 8, 10, 13, 15, 20, 25, 30, 35, 40]
Ni = [20, 25, 30, 40, 50, 35, 32, 25, 20, 12]
ni = [4, 6, 9, 14, 22, 18, 19, 16, 15, 10]

# Construct core dataframe schema
df_matrix = pd.DataFrame({
    'OBS': obs,
    'INCOME': income,
    'Ni': Ni,
    'ni': ni
})

# Sequential matrix computations with correct column keys
df_matrix['Pi = ni/Ni'] = df_matrix['ni'] / df_matrix['Ni']
df_matrix['1 - Pi'] = 1 - df_matrix['Pi = ni/Ni']
df_matrix['Pi / (1 - Pi)'] = df_matrix['Pi = ni/Ni'] / df_matrix['1 - Pi']
df_matrix['Li = ln(Pi/(1-Pi))'] = np.log(df_matrix['Pi / (1 - Pi)'])

# Reciprocal of error variance components (Gauss-Markov weight elements)
df_matrix['Wi = Ni*Pi*(1-Pi)'] = df_matrix['Ni'] * df_matrix['Pi = ni/Ni'] * df_matrix['1 - Pi']
df_matrix['sqrt(Wi)'] = np.sqrt(df_matrix['Wi'])
df_matrix['Li* = Li*sqrt(Wi)'] = df_matrix['Li = ln(Pi/(1-Pi))'] * df_matrix['sqrt(Wi)']
df_matrix['X* = Xi*sqrt(Wi)'] = df_matrix['INCOME'] * df_matrix['sqrt(Wi)']
df_matrix['Li*X*'] = df_matrix['Li* = Li*sqrt(Wi)'] * df_matrix['X* = Xi*sqrt(Wi)']
df_matrix['Li*sqrt(Wi)'] = df_matrix['Li* = Li*sqrt(Wi)'] * df_matrix['sqrt(Wi)']
df_matrix['X*sqrt(Wi)'] = df_matrix['X* = Xi*sqrt(Wi)'] * df_matrix['sqrt(Wi)']
df_matrix['X*^2'] = df_matrix['X* = Xi*sqrt(Wi)'] ** 2

avg_X = df_matrix['INCOME'].mean()
df_matrix['X - X_bar'] = df_matrix['INCOME'] - avg_X
df_matrix['(x - x_bar)^2'] = df_matrix['X - X_bar'] ** 2

avg_L = df_matrix['Li = ln(Pi/(1-Pi))'].mean()
df_matrix['L - L_bar'] = df_matrix['Li = ln(Pi/(1-Pi))'] - avg_L
df_matrix['(L - L_bar)*(x - x_bar)'] = df_matrix['L - L_bar'] * df_matrix['X - X_bar']

# Construct Summary Total Row safely
total_row = {col: np.nan for col in df_matrix.columns}
total_row['OBS'] = 'Total'
total_row['INCOME'] = float(sum(income))
total_row['Ni'] = float(sum(Ni))
total_row['ni'] = float(sum(ni))
total_row['Pi = ni/Ni'] = df_matrix['Pi = ni/Ni'].sum()
total_row['1 - Pi'] = df_matrix['1 - Pi'].sum()
total_row['Pi / (1 - Pi)'] = df_matrix['Pi / (1 - Pi)'].sum()
total_row['Li = ln(Pi/(1-Pi))'] = df_matrix['Li = ln(Pi/(1-Pi))'].sum()
total_row['Wi = Ni*Pi*(1-Pi)'] = df_matrix['Wi = Ni*Pi*(1-Pi)'].sum()
total_row['sqrt(Wi)'] = df_matrix['sqrt(Wi)'].sum()
total_row['Li* = Li*sqrt(Wi)'] = df_matrix['Li* = Li*sqrt(Wi)'].sum()
total_row['X* = Xi*sqrt(Wi)'] = df_matrix['X* = Xi*sqrt(Wi)'].sum()
total_row['Li*X*'] = df_matrix['Li*X*'].sum()
total_row['Li*sqrt(Wi)'] = df_matrix['Li*sqrt(Wi)'].sum()
total_row['X*sqrt(Wi)'] = df_matrix['X*sqrt(Wi)'].sum()
total_row['X*^2'] = df_matrix['X*^2'].sum()
total_row['X - X_bar'] = 0.0
total_row['(x - x_bar)^2'] = df_matrix['(x - x_bar)^2'].sum()
total_row['L - L_bar'] = df_matrix['L - L_bar'].sum()
total_row['(L - L_bar)*(x - x_bar)'] = df_matrix['(L - L_bar)*(x - x_bar)'].sum()

df_display = pd.concat([df_matrix, pd.DataFrame([total_row])], ignore_index=True)

# Model Fitting
X_matrix = sm.add_constant(df_matrix['INCOME'])
model_ols = sm.OLS(df_matrix['Li = ln(Pi/(1-Pi))'], X_matrix).fit()
model_wls = sm.WLS(df_matrix['Li = ln(Pi/(1-Pi))'], X_matrix, weights=df_matrix['Wi = Ni*Pi*(1-Pi)']).fit()

def build_coef_table(model):
    return pd.DataFrame({
        'Coefficient': model.params.round(4),
        'Std. Error': model.bse.round(4),
        't-Statistic': model.tvalues.round(4),
        'P-Value': model.pvalues.round(4)
    })

ols_coef_df = build_coef_table(model_ols)
wls_coef_df = build_coef_table(model_wls)

# Key analytical values calculated via OLS parameters
prob_mean = 1 / (1 + np.exp(-(model_ols.params['const'] + model_ols.params['INCOME'] * avg_X)))
odds = np.exp(model_ols.params['INCOME'])
m_effect = model_ols.params['INCOME'] * prob_mean * (1 - prob_mean)

# Interactive custom prediction panel widget setup
st.sidebar.header("🔮 Custom Scenario Simulator")
user_income_input = st.sidebar.slider("Select Custom Income Level ($X_i$):", min_value=1.0, max_value=60.0, value=float(avg_X), step=0.5)
sim_model = st.sidebar.selectbox("Choose Estimation Parameters:", ["WLS (Heteroscedasticity Corrected)", "OLS (Unweighted)"])

selected_params = model_wls.params if sim_model == "WLS (Heteroscedasticity Corrected)" else model_ols.params
pred_log_odds = selected_params['const'] + selected_params['INCOME'] * user_income_input
pred_prob = 1 / (1 + np.exp(-pred_log_odds))
pred_marginal = selected_params['INCOME'] * pred_prob * (1 - pred_prob)

st.sidebar.metric(label="Predicted Log-Odds", value=f"{pred_log_odds:.4f}")
st.sidebar.metric(label="Probability of Car Ownership", value=f"{pred_prob * 100:.2f}%")
st.sidebar.metric(label="Marginal Effect at this Level", value=f"{pred_marginal:.4f}")

# Layout columns for data presentation
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 Table 1.1: Complete Comprehensive Econometric Transformation Matrix")
    def safe_float_formatter(val):
        if isinstance(val, (float, np.float64, int, np.int64)):
            return f"{val:.4f}" if not pd.isna(val) else ""
        return str(val)
    st.dataframe(df_display.style.format(safe_float_formatter), use_container_width=True)

with col2:
    st.header("📥 Research Deliverables")
    paper_content = f"""MICROECONOMETRICS RESEARCH PAPER: LOGISTIC VEHICLE OWNERSHIP MODEL
======================================================================
1. LOGIT MODEL ESTIMATION: L_i = {model_ols.params['const']:.4f} + {model_ols.params['INCOME']:.4f} * X
2. ODDS RATIO ANALYSIS: e^(b1) = {odds:.4f}
3. PREDICTED PROBABILITY AT MEAN INCOME: {prob_mean * 100:.2f}%
4. MARGINAL EFFECT AT MEAN INCOME: {m_effect:.4f}
5. WEIGHTED LEAST SQUARES HETEROSCEDASTICITY ADJUSTMENT
"""
    st.download_button(label="Click Here to Download Full Paper (.txt)", data=paper_content, file_name="microeconometrics_car_ownership_paper.txt", mime="text/plain")

st.markdown("---")

# Visual chart layer
st.header("📈 Visualizing the Estimated S-Curve Function vs. Observed Probabilities")
x_curve = np.linspace(0, 50, 200)
y_curve_ols = 1 / (1 + np.exp(-(model_ols.params['const'] + model_ols.params['INCOME'] * x_curve)))
y_curve_wls = 1 / (1 + np.exp(-(model_wls.params['const'] + model_wls.params['INCOME'] * x_curve)))

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_matrix['INCOME'], y=df_matrix['Pi = ni/Ni'], mode='markers', name='Observed (ni/Ni)', marker=dict(size=df_matrix['Ni']/2, color='darkred', line=dict(width=1))))
fig.add_trace(go.Scatter(x=x_curve, y=y_curve_wls, mode='lines', name='WLS Fitted Curve', line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=x_curve, y=y_curve_ols, mode='lines', name='OLS Fitted Curve', line=dict(color='gray', width=2, dash='dash')))
fig.add_trace(go.Scatter(x=[user_income_input], y=[pred_prob], mode='markers', name='Simulated Point', marker=dict(color='gold', size=15, symbol='star')))

fig.update_layout(
    font=dict(family="Times New Roman", size=14),
    xaxis_title="Monthly Income Level (X)",
    yaxis_title="Probability of Vehicle Ownership (P)",
    legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("🧮 Model Estimations (Full Regression Summary)")
model_type = st.radio("Select Model Specification to View:", ["Ordinary Least Squares (OLS)", "Weighted Least Squares (WLS)"])
if model_type == "Ordinary Least Squares (OLS)":
    st.text(str(model_ols.summary().tables[1]))
else:
    st.text(str(model_wls.summary().tables[1]))

st.markdown("---")
st.header("📝 Microeconometric Interpretations & Analysis Guide")

st.markdown("### 1. Logit Model Estimation & Direction")
st.latex(r"L_i = \ln\left(\frac{p_i}{1 - p_i}\right) = \beta_0 + \beta_1 X_i + u_i")
st.markdown("**Estimated Equation (OLS vs WLS):**")
st.latex(rf"\text{{OLS: }} \hat{{L_i}} = {model_ols.params['const']:.4f} + {model_ols.params['INCOME']:.4f} X")
st.latex(rf"\text{{WLS: }} \hat{{L_i}} = {model_wls.params['const']:.4f} + {model_wls.params['INCOME']:.4f} X")
st.dataframe(wls_coef_df, use_container_width=True)

st.markdown("### 2. Odds Ratio Analysis")
st.latex(r"\text{Odds Ratio} = e^{\beta_1}")
st.metric(label="Calculated WLS Odds Ratio (e^b1)", value=f"{np.exp(model_wls.params['INCOME']):.4f}")
st.write(f"For every additional unit increase in income, the odds of owning a car scale by a factor of {np.exp(model_wls.params['INCOME']):.4f} under efficient estimation.")
