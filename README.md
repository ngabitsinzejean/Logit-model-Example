# Interactive Logistic Car Ownership Dashboard 📊

An enterprise-grade microeconometrics dashboard built to evaluate binary choice frameworks using grouped household data. This platform compares standard Ordinary Least Squares (OLS) trajectories against heteroscedasticity-corrected Weighted Least Squares (WLS) models using empirical logits.

## 🚀 Live Application Link
Access the public interactive application here: [Your Streamlit App URL Placement]

## 🧮 Econometric Methodology Implemented
1. **Empirical Probability Transformation:** Calculates cell-by-cell probabilities ($p_i = n_i / N_i$).
2. **Logit Linearisation:** Transforms categorical data into continuous space using empirical log-odds ($L_i = \ln(p_i / (1 - p_i))$).
3. **Heteroscedasticity Correction:** Computes analytical matrix weights based on theoretical error variance ($\text{Var}(u_i) = 1 / [N_i p_i (1 - p_i)]$).
4. **WLS Estimation:** Applies Weighted Least Squares to ensure execution satisfies Gauss-Markov properties for BLUE estimators.

## 📦 Core Technology Stack
- **Frontend Framework:** Streamlit
- **Statistical Processing Engine:** Statsmodels (OLS & WLS modules)
- **Data Engineering:** Pandas & NumPy
- **Layout & Typographic Enforcements:** Embedded CSS (Times New Roman 12pt Standard)

## 📥 Deployment Specifications
This repository includes a standalone deployment setup optimized for containerized cloud nodes (Streamlit Cloud, Render, or Heroku architectures).
