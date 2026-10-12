import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Microeconometrics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("📊 Microeconometrics Dashboard")
st.subheader("Logistic Car Ownership Model")
st.write(
    "An interactive evaluation of binary choice frameworks "
    "for grouped data."
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Settings")

beta0 = st.sidebar.number_input(
    "β₀ (Intercept)",
    value=-493.54,
    format="%.4f"
)

beta1 = st.sidebar.number_input(
    "β₁ (Coefficient of X)",
    value=32.96,
    format="%.4f"
)

st.sidebar.markdown("---")

st.sidebar.write("### Logistic Model")

st.sidebar.latex(
    r"""
    P_i=\frac{1}{1+e^{-L_i}}
    """
)

st.sidebar.latex(
    r"""
    L_i=\beta_0+\beta_1X_i
    """
)

# ============================================================
# DATA INPUT
# ============================================================

st.header("1. Data")

st.write(
    "Enter the explanatory-variable values and the observed "
    "binary car-ownership outcome."
)

# Default example data
default_data = pd.DataFrame({
    "Family": range(1, 11),
    "X": [
        14.5,
        14.8,
        15.0,
        15.2,
        15.5,
        15.8,
        16.0,
        16.2,
        16.5,
        16.8
    ],
    "Y": [
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        1
    ]
})

df = st.data_editor(
    default_data,
    num_rows="dynamic",
    use_container_width=True
)

# ============================================================
# CHECK DATA
# ============================================================

if "X" not in df.columns:
    st.error("The dataset must contain a column named X.")
    st.stop()

if "Y" not in df.columns:
    st.error("The dataset must contain a column named Y.")
    st.stop()

# Convert variables to numeric
df["X"] = pd.to_numeric(df["X"], errors="coerce")
df["Y"] = pd.to_numeric(df["Y"], errors="coerce")

# Remove rows with missing values
df = df.dropna(subset=["X", "Y"]).copy()

# ============================================================
# LOGISTIC CALCULATIONS
# ============================================================

df_matrix = df.copy()

# ------------------------------------------------------------
# Linear predictor
# Li = β0 + β1Xi
# ------------------------------------------------------------

df_matrix["Li"] = (
    beta0 +
    beta1 * df_matrix["X"]
)

# ------------------------------------------------------------
# Predicted probability
# Pi = 1 / (1 + exp(-Li))
# ------------------------------------------------------------

df_matrix["Pi"] = (
    1 /
    (
        1 +
        np.exp(
            -np.clip(
                df_matrix["Li"],
                -700,
                700
            )
        )
    )
)

# ------------------------------------------------------------
# 1 - Pi
# ------------------------------------------------------------

df_matrix["1 - Pi"] = (
    1 - df_matrix["Pi"]
)

# ------------------------------------------------------------
# Wi = Pi(1-Pi)
# ------------------------------------------------------------

df_matrix["Wi"] = (
    df_matrix["Pi"] *
    df_matrix["1 - Pi"]
)

# ------------------------------------------------------------
# sqrt(Wi)
# ------------------------------------------------------------

df_matrix["sqrt(Wi)"] = np.sqrt(
    df_matrix["Wi"]
)

# ------------------------------------------------------------
# Li * sqrt(Wi)
# ------------------------------------------------------------

df_matrix["Li * sqrt(Wi)"] = (
    df_matrix["Li"] *
    df_matrix["sqrt(Wi)"]
)

# ============================================================
# RESULTS
# ============================================================

st.header("2. Logistic Model Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Observations",
        len(df_matrix)
    )

with col2:
    st.metric(
        "Mean Probability",
        f"{df_matrix['Pi'].mean():.4f}"
    )

with col3:
    st.metric(
        "Minimum Probability",
        f"{df_matrix['Pi'].min():.4f}"
    )

with col4:
    st.metric(
        "Maximum Probability",
        f"{df_matrix['Pi'].max():.4f}"
    )

# ============================================================
# CALCULATION TABLE
# ============================================================

st.header("3. Logistic Probability Calculation Table")

display_columns = [
    "Family",
    "X",
    "Y",
    "Li",
    "Pi",
    "1 - Pi",
    "Wi",
    "sqrt(Wi)",
    "Li * sqrt(Wi)"
]

# Only display columns that actually exist
display_columns = [
    col for col in display_columns
    if col in df_matrix.columns
]

st.dataframe(
    df_matrix[display_columns].style.format(
        {
            "X": "{:.4f}",
            "Li": "{:.4f}",
            "Pi": "{:.6f}",
            "1 - Pi": "{:.6f}",
            "Wi": "{:.6f}",
            "sqrt(Wi)": "{:.6f}",
            "Li * sqrt(Wi)": "{:.6f}"
        }
    ),
    use_container_width=True
)

# ============================================================
# FORMULAS
# ============================================================

st.header("4. Mathematical Framework")

st.markdown("### Linear Predictor")

st.latex(
    r"""
    L_i=\beta_0+\beta_1X_i
    """
)

st.markdown("### Predicted Probability")

st.latex(
    r"""
    P_i=\frac{1}{1+e^{-L_i}}
    """
)

st.markdown("### Complementary Probability")

st.latex(
    r"""
    1-P_i
    """
)

st.markdown("### Logistic Variance")

st.latex(
    r"""
    W_i=P_i(1-P_i)
    """
)

st.markdown("### Square Root of the Weight")

st.latex(
    r"""
    \sqrt{W_i}=\sqrt{P_i(1-P_i)}
    """
)

# ============================================================
# INTERPRETATION
# ============================================================

st.header("5. Interpretation")

st.write(
    """
    The linear predictor Li represents the estimated latent index
    associated with car ownership. The logistic transformation
    converts this index into a probability between 0 and 1.
    """
)

st.write(
    """
    Pi is the predicted probability that household i owns a car.
    Therefore, 1 − Pi is the predicted probability that the
    household does not own a car.
    """
)

st.write(
    """
    Wi = Pi(1 − Pi) represents the variance component used in
    logistic regression. It reaches its highest value when the
    predicted probability is close to 0.5 and becomes smaller
    when the probability approaches 0 or 1.
    """
)

# ============================================================
# CLASSIFICATION
# ============================================================

st.header("6. Predicted Car Ownership")

threshold = st.slider(
    "Classification threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

df_matrix["Predicted Y"] = (
    df_matrix["Pi"] >= threshold
).astype(int)

classification_columns = [
    "Family",
    "X",
    "Y",
    "Pi",
    "Predicted Y"
]

classification_columns = [
    col for col in classification_columns
    if col in df_matrix.columns
]

st.dataframe(
    df_matrix[classification_columns],
    use_container_width=True
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.header("7. Classification Performance")

actual = df_matrix["Y"]
predicted = df_matrix["Predicted Y"]

TP = ((actual == 1) & (predicted == 1)).sum()
TN = ((actual == 0) & (predicted == 0)).sum()
FP = ((actual == 0) & (predicted == 1)).sum()
FN = ((actual == 1) & (predicted == 0)).sum()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("True Positive", TP)

with c2:
    st.metric("True Negative", TN)

with c3:
    st.metric("False Positive", FP)

with c4:
    st.metric("False Negative", FN)

# ============================================================
# ACCURACY
# ============================================================

if len(df_matrix) > 0:
    accuracy = (
        (TP + TN) /
        len(df_matrix)
    )

    st.metric(
        "Classification Accuracy",
        f"{accuracy:.2%}"
    )

# ============================================================
# PROBABILITY GRAPH
# ============================================================

st.header("8. Predicted Probability")

fig, ax = plt.subplots()

ax.scatter(
    df_matrix["X"],
    df_matrix["Pi"]
)

ax.set_xlabel("X")
ax.set_ylabel("Predicted Probability (Pi)")
ax.set_title("Logistic Predicted Probability")

ax.axhline(
    threshold,
    linestyle="--"
)

st.pyplot(fig)

# ============================================================
# LOGISTIC CURVE
# ============================================================

st.header("9. Logistic Curve")

x_min = df_matrix["X"].min()
x_max = df_matrix["X"].max()

if x_min == x_max:
    x_curve = np.linspace(
        x_min - 1,
        x_max + 1,
        200
    )
else:
    x_curve = np.linspace(
        x_min,
        x_max,
        200
    )

l_curve = (
    beta0 +
    beta1 * x_curve
)

p_curve = (
    1 /
    (
        1 +
        np.exp(
            -np.clip(
                l_curve,
                -700,
                700
            )
        )
    )
)

fig2, ax2 = plt.subplots()

ax2.plot(
    x_curve,
    p_curve
)

ax2.axhline(
    0.5,
    linestyle="--"
)

ax2.set_xlabel("X")
ax2.set_ylabel("P(Y = 1 | X)")
ax2.set_title("Logistic Probability Curve")

st.pyplot(fig2)

# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.header("10. Download Results")

csv = df_matrix.to_csv(
    index=False
)

st.download_button(
    label="📥 Download Results as CSV",
    data=csv,
    file_name="logistic_car_ownership_results.csv",
    mime="text/csv"
)

# ============================================================
# CONCLUSION
# ============================================================

st.header("11. Conclusion")

st.success(
    """
    The logistic model estimates the probability of car ownership
    for each household. The model first calculates the linear
    predictor Li, then transforms Li into the predicted probability
    Pi. The quantities 1 − Pi, Wi, sqrt(Wi), and Li × sqrt(Wi)
    are subsequently calculated from Pi.
    """
)
