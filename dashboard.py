import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Set application layout and page config
st.set_page_config(page_title="Microeconometrics: Logistic Car Ownership Model", layout="wide")

# GLOBAL TYPOGRAPHY LAYER: Enforces Times New Roman 14pt (18.5px) globally across text fields
st.markdown(
    """
    <style>
    /* Target paragraphs, items, data selectors, and custom markdown text blocks */
    p, li, span, label, .stMarkdown p, .stVerticalBlock div p {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 18.5px !important; /* 18.5px is equivalent to 14pt font size */
    }
   
    /* Responsive adjustment for headers using Times New Roman style */
    .stMarkdown h1, h1 { font-family: 'Times New Roman', serif !important; font-size: 26px !important; font-weight: bold !important; }
    .stMarkdown h2, h2 { font-family: 'Times New Roman', serif !important; font-size: 22px !important; font-weight: bold !important; }
    .stMarkdown h3, h3 { font-family: 'Times New Roman', serif !important; font-size: 20px !important; font-weight: bold !important; }
    
    /* Ensure DataFrame values are highly legible while remaining structured */
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

# Sequential matrix matrix computations
df_matrix['Pi = ni/Ni'] = df_matrix['ni'] / df_matrix['Ni']
df_matrix['1 - Pi'] = 1 - df_matrix['Pi']
df_matrix['Pi / (1 - Pi)'] = df_matrix['Pi'] / df_matrix['1 - Pi']
df_matrix['Li = ln(Pi/(1-Pi))'] = np.log(df_matrix['Pi / (1 - Pi)'])
df_matrix['Wi = Ni*Pi*(1-Pi)'] = df_matrix['Ni'] * df_matrix['Pi'] * df_matrix['1 - Pi']
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

# Construct the Summary Total row aggregation values
total_row = {col: '' for col in df_matrix.columns}
total_row['OBS'] = 'Total'
total_row['INCOME'] = sum(income)
total_row['Ni'] = sum(Ni)
total_row['ni'] = sum(ni)
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

# Process analytical statistical modules
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

prob_mean = 1 / (1 + np.exp(-(model_ols.params['const'] + model_ols.params['INCOME'] * avg_X)))
odds = np.exp(model_ols.params['INCOME'])
m_effect = model_ols.params['INCOME'] * prob_mean * (1 - prob_mean)

# Generate report string mapping
paper_content = f"""MICROECONOMETRICS RESEARCH PAPER: LOGISTIC VEHICLE OWNERSHIP MODEL
======================================================================
1. LOGIT MODEL ESTIMATION: L_i = {model_ols.params['const']:.4f} + {model_ols.params['INCOME']:.4f} * X
2. ODDS RATIO ANALYSIS: e^(b1) = {odds:.4f}
3. PREDICTED PROBABILITY AT MEAN INCOME: {prob_mean * 100:.2f}%
4. MARGINAL EFFECT AT MEAN INCOME: {m_effect:.4f}
5. WEIGHTED LEAST SQUARES HETEROSCEDASTICITY ADJUSTMENT ADJUSTMENT
"""

st.header("📥 Download Complete Research Paper")
st.download_button(label="Click Here to Download Full Paper (.txt)", data=paper_content, file_name="microeconometrics_car_ownership_paper.txt", mime="text/plain")
st.markdown("---")

# Render comprehensive calculation data array sheet
st.header("📋 Table 1.1: Complete Comprehensive Econometric Transformation Matrix")
st.dataframe(df_display.style.format(lambda x: f"{x:.4f}" if isinstance(x, (float, np.float64)) else str(x)), width='stretch')
st.markdown("---")

st.header("🧮 Model Estimations (Full Regression Summary)")
model_type = st.radio("Select Model Specification to View:", ["Ordinary Least Squares (OLS)", "Weighted Least Squares (WLS)"])
if model_type == "Ordinary Least Squares (OLS)":
    st.text(str(model_ols.summary().tables))
else:
    st.text(str(model_wls.summary().tables))

st.markdown("---")
st.header("📝 Microeconometric Interpretations & Analysis Guide")

# Structured sequential interpretation guide segments
st.markdown("### 1. Logit Model Estimation & Direction")
st.markdown("**Theoretical Formula:**")
st.latex(r"L_i = \ln\left(\frac{p_i}{1 - p_i}\right) = \beta_0 + \beta_1 X_i + u_i")
st.markdown("**Estimated Equation:**")
st.latex(rf"\hat{{L_i}} = {model_ols.params['const']:.4f} + {model_ols.params['INCOME']:.4f} X")
st.markdown("**Table of Coefficients (OLS):**")
st.dataframe(ols_coef_df, width='stretch')
st.write("There is a positive relationship between monthly wage and car ownership. As monthly income increases, the log-odds of vehicle ownership increases.")
st.markdown("---")

st.markdown("### 2. Odds Ratio Analysis")
st.markdown("**Theoretical Formula:**")
st.latex(r"\text{Odds Ratio} = e^{\beta_1}")
st.metric(label="Calculated Odds Ratio (e^b1)", value=f"{odds:.4f}")
st.write(f"For every 1-unit increase in income, the odds of owning a car grow by a factor of {odds:.4f}.")
st.markdown("---")

st.markdown("### 3. Predicted Probability at Community Mean")
st.markdown("**Theoretical Formula:**")
st.latex(r"P(Y_i = 1) = \frac{1}{1 + e^{-\hat{L_i}}}")
st.write(f"Average Community Income Level: **{avg_X:.2f}**")
st.metric(label="Probability at Mean Income", value=f"{prob_mean * 100:.2f}%")
st.markdown("---")

st.markdown("### 4. Change in Probability (Marginal Effect)")
st.markdown("**Theoretical Formula:**")
st.latex(r"\text{Marginal Effect} = \beta_1 \cdot P_i (1 - P_i)")
st.metric(label="Marginal Effect at Mean", value=f"{m_effect:.4f}")
st.write(f"At the average income level, a 1-unit increase in income increases the probability of car ownership by {m_effect * 100:.2f} percentage points.")
st.markdown("---")

st.markdown("### 5. Heteroscedasticity & Weighted Least Squares")
st.markdown("**Theoretical Error Variance Formula:**")
st.latex(r"\text{Var}(u_i) = \frac{1}{N_i p_i (1 - p_i)}")
st.markdown("**Corrected Estimated WLS Coefficients:**")
st.dataframe(wls_coef_df, width='stretch')
st.write("Econometric Note: Grouped data logit models suffer from inherent heteroscedasticity because the variance of the error term changes across rows based on sample sizes and true probabilities. Weighting the entire equation by the reciprocal of this variance stabilizes the error variance, satisfying the standard Gauss-Markov assumptions for Blue estimators.")
