import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Set application layout and page config
st.set_page_config(page_title="Microeconometrics: Logistic Car Ownership Model", layout="wide")

# Refined CSS targeting text elements safely without corrupting internal icon strings
st.markdown(
    """
    <style>
    /* target main text blocks, headers, expander text, and dataframes */
    html, body, p, span, label, stDataFrame, .stMarkdown, .stExpander p, .stRadio label {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 16px !important; /* 16px equals 12pt font size */
    }
    /* target headers specifically while maintaining font consistency */
    h1 { font-family: 'Times New Roman', serif !important; font-size: 24px !important; font-weight: bold !important; }
    h2 { font-family: 'Times New Roman', serif !important; font-size: 20px !important; font-weight: bold !important; }
    h3 { font-family: 'Times New Roman', serif !important; font-size: 18px !important; font-weight: bold !important; }
    
    /* target dataframes specifically to ensure table headers use the correct font */
    .stDataFrame div {
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    /* Keep system code blocks clean and readable */
    pre, code {
        font-family: monospace !important;
        font-size: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Microeconometrics Dashboard: Logistic Car Ownership Model")
st.subheader("An Interactive Evaluation of Binary Choice Frameworks for Grouped Data")
st.markdown("---")

# Clean dataset initialization
X_data = [int(x) for x in "6,8,10,13,15,20,25,30,35,40".split(",")]
Ni_data = [int(x) for x in "20,25,30,40,50,35,32,25,20,12".split(",")]
ni_data = [int(x) for x in "4,6,9,14,22,18,19,16,15,10".split(",")]

df = pd.DataFrame({'Monthly Wage (X)': X_data, 'Total Households (Ni)': Ni_data, 'Car Owners (ni)': ni_data})

# Mathematical Transformations
df['Empirical Prob (pi)'] = df['Car Owners (ni)'] / df['Total Households (Ni)']
df['Empirical Logit (Li)'] = np.log(df['Car Owners (ni)'] / (df['Total Households (Ni)'] - df['Car Owners (ni)']))
df['WLS Weights (wi)'] = df['Total Households (Ni)'] * df['Empirical Prob (pi)'] * (1 - df['Empirical Prob (pi)'])

# Run models
X_matrix = sm.add_constant(df['Monthly Wage (X)'])
model_ols = sm.OLS(df['Empirical Logit (Li)'], X_matrix).fit()
model_wls = sm.WLS(df['Empirical Logit (Li)'], X_matrix, weights=df['WLS Weights (wi)']).fit()

# Convert statsmodels coefficient tables to clear pandas dataframes for layout styling
ols_coef_df = pd.read_html(model_ols.summary().tables[1].as_html(), header=0, index_col=0)[0]
wls_coef_df = pd.read_html(model_wls.summary().tables[1].as_html(), header=0, index_col=0)[0]

# Calculations for text report generation
avg_wage = df['Monthly Wage (X)'].mean()
p_logit = model_ols.params['const'] + model_ols.params['Monthly Wage (X)'] * avg_wage
prob_mean = 1 / (1 + np.exp(-p_logit))
odds = np.exp(model_ols.params['Monthly Wage (X)'])
m_effect = model_ols.params['Monthly Wage (X)'] * prob_mean * (1 - prob_mean)

# Prepare full paper text for download file
paper_content = f"""MICROECONOMETRICS RESEARCH PAPER: LOGISTIC VEHICLE OWNERSHIP MODEL
======================================================================
Font Specification: Times New Roman 12pt
Framework: Binary Choice Model for Grouped Data

--- PART (i): LOGIT MODEL ESTIMATION ---
Theoretical Formula: L_i = ln(p_i / (1 - p_i)) = b0 + b1*X_i + u_i
Estimated Equation: L_i_hat = {model_ols.params['const']:.4f} + {model_ols.params['Monthly Wage (X)']:.4f} * X
Interpretation: There is a positive relationship between monthly wage and car ownership. 
As monthly income increases, the log-odds of a household owning a vehicle increases by {model_ols.params['Monthly Wage (X)']:.4f} units for every additional unit of wage.

--- PART (ii): ODDS RATIO ANALYSIS ---
Theoretical Formula: Odds Ratio = e^(b1)
Calculated Odds Ratio: {odds:.4f}
Interpretation: For every 1-unit increase in monthly wage, the odds of a household owning a vehicle increase by a factor of {odds:.4f} (or roughly {((odds - 1) * 100):.2f}%).

--- PART (iii): PREDICTED PROBABILITY AT COMMUNITY MEAN ---
Theoretical Formula: P_i = 1 / (1 + e^(-L_i))
Average Community Wage Level (X_bar): {avg_wage:.2f}
Predicted Probability: {prob_mean * 100:.2f}%

--- PART (iv): MARGINAL EFFECT ---
Theoretical Formula: Marginal Effect = b1 * P_i * (1 - P_i)
Calculated Marginal Effect: {m_effect:.4f}
Interpretation: At the average income level, a 1-unit increase in monthly wage leads to an increase of approximately {m_effect * 100:.2f} percentage points in the actual probability of vehicle ownership.

--- PART (v): HETEROSCEDASTICITY & WEIGHTED LEAST SQUARES ---
Theoretical Error Variance Formula: Var(u_i) = 1 / [N_i * p_i * (1 - p_i)]
Corrected WLS Equation: L_i_hat* = {model_wls.params['const']:.4f} + {model_wls.params['Monthly Wage (X)']:.4f} * X
Econometric Note: Grouped data logit models suffer from inherent heteroscedasticity because the variance of the error term changes across groups based on sample sizes and true cell probabilities. Weighting the entire equation by the reciprocal of this variance stabilizes the error variances, satisfying the standard Gauss-Markov assumptions for Best Linear Unbiased Estimators (BLUE).

======================================================================
COMPLETE COEFFICIENTS AND REGRESSION SUMMARY TABLES
======================================================================
1. OLS MODEL TABLES:
{str(model_ols.summary().tables[1])}

2. WLS MODEL TABLES:
{str(model_wls.summary().tables[1])}
======================================================================
"""

# Download button component
st.header("📥 Download Complete Research Paper")
st.download_button(
    label="Click Here to Download Full Paper (.txt)",
    data=paper_content,
    file_name="microeconometrics_car_ownership_paper.txt",
    mime="text/plain"
)
st.markdown("---")

# Full-width display for transformed data table
st.header("📋 Table 1.1: Transformed Grouped Data")
st.dataframe(df.style.format(precision=4), width='stretch')
st.markdown("---")

# Full-width display for regression outputs
st.header("🧮 Model Estimations (Full Regression Summary)")
model_type = st.radio("Select Model Specification to View:", ["Ordinary Least Squares (OLS)", "Weighted Least Squares (WLS)"])

if model_type == "Ordinary Least Squares (OLS)":
    st.text("OLS Regression Summary Table:")
    st.text(str(model_ols.summary().tables[0]))
    st.text(str(model_ols.summary().tables[1]))
else:
    st.text("WLS Heteroscedasticity Corrected Summary Table:")
    st.text(str(model_wls.summary().tables[0]))
    st.text(str(model_wls.summary().tables[1]))

st.markdown("---")
st.header("📝 Microeconometric Interpretations & Analysis Guide")

with st.expander("(i) Logit Model Estimation & Direction"):
    st.markdown("**Theoretical Formula:**")
    st.latex(r"L_i = \ln\left(\frac{p_i}{1 - p_i}\right) = \beta_0 + \beta_1 X_i + u_i")
    st.markdown("**Estimated Equation:**")
    st.latex(rf"\hat{{L_i}} = {model_ols.params['const']:.4f} + {model_ols.params['Monthly Wage (X)']:.4f} X")
    st.markdown("**Table of Coefficients (OLS):**")
    st.dataframe(ols_coef_df, width='stretch')
    st.write(f"**Interpretation:** There is a **positive relationship** between monthly wage and car ownership. As monthly income increases, the log-odds of a household owning a vehicle increases by {model_ols.params['Monthly Wage (X)']:.4f} units for every additional unit of wage.")

with st.expander("(ii) Odds Ratio Analysis"):
    st.markdown("**Theoretical Formula:**")
    st.latex(r"\text{Odds Ratio} = \frac{\text{Odds}(X + 1)}{\text{Odds}(X)} = e^{\beta_1}")
    st.metric(label="Calculated Odds Ratio (e^b1)", value=f"{odds:.4f}")
    st.write(f"**Interpretation:** For every 1-unit increase in monthly wage, the odds of a household owning a vehicle increase by a factor of **{odds:.4f}** (which means the odds grow by roughly {((odds - 1) * 100):.2f}%).")

with st.expander("(iii) Predicted Probability at Community Mean"):
    st.markdown("**Theoretical Formula:**")
    st.latex(r"P(Y_i = 1 \mid X_i) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_i)}}")
    st.write(f"Average Community Wage Level (\overline{{X}}): **{avg_wage:.2f}**")
    st.metric(label="Probability of Car Ownership", value=f"{prob_mean * 100:.2f}%")

with st.expander("(iv) Change in Probability (Marginal Effect)"):
    st.markdown("**Theoretical Formula:**")
    st.latex(r"\text{Marginal Effect} = \frac{\partial P_i}{\partial X_i} = \beta_1 \cdot P_i (1 - P_i)")
    st.metric(label="Marginal Effect at Mean", value=f"{m_effect:.4f}")
    st.write(f"**Interpretation:** At the average income level, a 1-unit increase in monthly wage leads to an increase of approximately **{m_effect * 100:.2f} percentage points** in the actual probability of vehicle ownership.")

with st.expander("(v) Heteroscedasticity & Weighted Least Squares"):
    st.markdown("**Theoretical Error Variance Formula:**")
    st.latex(r"\text{Var}(u_i) = \frac{1}{N_i p_i (1 - p_i)}")
    st.markdown("**Corrected Estimated WLS Equation:**")
    st.latex(rf"\hat{{L_i}}^* = {model_wls.params['const']:.4f} + {model_wls.params['Monthly Wage (X)']:.4f} X")
    st.markdown("**Table of Coefficients (Corrected WLS):**")
    st.dataframe(wls_coef_df, width='stretch')
    st.write("**Econometric Note:** Grouped data logit models suffer from inherent heteroscedasticity because the variance of the error term changes across rows based on sample sizes and true probabilities. Weighting the entire equation by the reciprocal of this variance stabilizes the error variance, satisfying the standard Gauss-Markov assumptions for Blue estimators.")
