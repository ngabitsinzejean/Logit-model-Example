import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Set application layout and page config
st.set_page_config(page_title="Microeconometrics: Logistic Car Ownership Model", layout="wide")

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

# Layout splits
col1, col2 = st.columns(2)

with col1:
    st.header("📋 Table 1.1: Transformed Grouped Data")
    st.dataframe(df.style.format(precision=4), width='stretch')
    
    st.header("🧮 Model Estimations (Table of Coefficients)")
    model_type = st.radio("Select Model Specification to View:", ["Ordinary Least Squares (OLS)", "Weighted Least Squares (WLS)"])
    
    if model_type == "Ordinary Least Squares (OLS)":
        st.text("OLS Regression Summary Table:")
        st.text(str(model_ols.summary().tables))
    else:
        st.text("WLS Heteroscedasticity Corrected Summary Table:")
        st.text(str(model_wls.summary().tables))

with col2:
    st.header("📈 Interactive Data Visualizations")
    viz_choice = st.selectbox("Choose Chart Type:", ["Log-Odds Linear Trajectory", "S-Curve Probability Fit"])
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    if viz_choice == "Log-Odds Linear Trajectory":
        ax.scatter(df['Monthly Wage (X)'], df['Empirical Logit (Li)'], color='darkred', label='Observed Logits', zorder=3)
        wage_domain = np.linspace(5, 45, 100)
        ols_lines = model_ols.params['const'] + model_ols.params['Monthly Wage (X)'] * wage_domain
        wls_lines = model_wls.params['const'] + model_wls.params['Monthly Wage (X)'] * wage_domain
        ax.plot(wage_domain, ols_lines, color='blue', linestyle='--', label='OLS Estimate')
        ax.plot(wage_domain, wls_lines, color='green', label='WLS Estimate (Corrected)')
        ax.set_ylabel("Empirical Logit (Log-Odds)")
        ax.set_title("Linear Fit of Log-Odds vs Monthly Wage")
    else:
        ax.scatter(df['Monthly Wage (X)'], df['Empirical Prob (pi)'], color='indigo', label='Observed Probabilities', zorder=3)
        wage_domain = np.linspace(5, 45, 100)
        logit_pred = model_ols.params['const'] + model_ols.params['Monthly Wage (X)'] * wage_domain
        prob_pred = 1 / (1 + np.exp(-logit_pred))
        ax.plot(wage_domain, prob_pred, color='darkorange', linewidth=2, label='Fitted Logit S-Curve')
        ax.set_ylabel("Probability of Owning a Vehicle (pi)")
        ax.set_title("Fitted Logit Probability S-Curve Graph")
        
    ax.set_xlabel("Monthly Wage (X)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
st.header("📝 Microeconometric Interpretations & Analysis Guide")

with st.expander("(i) Logit Model Estimation & Direction"):
    st.latex(rf"\hat{{L_i}} = {model_ols.params['const']:.4f} + {model_ols.params['Monthly Wage (X)']:.4f} X")
    st.write(f"**Interpretation:** Positive relationship between monthly wage and car ownership. Log-odds increase by {model_ols.params['Monthly Wage (X)']:.4f} units per unit of wage.")

with st.expander("(ii) Odds Ratio Analysis"):
    odds = np.exp(model_ols.params['Monthly Wage (X)'])
    st.metric(label="Calculated Odds Ratio (e^b1)", value=f"{odds:.4f}")
    st.write(f"**Interpretation:** For every 1-unit increase in monthly wage, the odds of owning a vehicle increase by a factor of **{odds:.4f}**.")

with st.expander("(iii) Predicted Probability at Community Mean"):
    avg_wage = df['Monthly Wage (X)'].mean()
    p_logit = model_ols.params['const'] + model_ols.params['Monthly Wage (X)'] * avg_wage
    prob_mean = 1 / (1 + np.exp(-p_logit))
    st.write(f"Average Community Wage Level: **{avg_wage:.2f}**")
    st.metric(label="Probability of Car Ownership", value=f"{prob_mean * 100:.2f}%")

with st.expander("(iv) Change in Probability (Marginal Effect)"):
    m_effect = model_ols.params['Monthly Wage (X)'] * prob_mean * (1 - prob_mean)
    st.metric(label="Marginal Effect at Mean", value=f"{m_effect:.4f}")
    st.write(f"**Interpretation:** At the average income level, a 1-unit increase in monthly wage leads to an increase of approximately **{m_effect * 100:.2f} percentage points** in probability.")

with st.expander("(v) Heteroscedasticity & Weighted Least Squares"):
    st.latex(rf"\hat{{L_i}} = {model_wls.params['const']:.4f} + {model_wls.params['Monthly Wage (X)']:.4f} X")
    st.write("**Econometric Note:** Grouped data logit models suffer from heteroscedasticity because var(ui) = 1 / [Ni * pi * (1 - pi)]. Weighting by the reciprocal stabilizes the errors.")
