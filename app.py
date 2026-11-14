import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="CorroPredict AI", page_icon="🧪", layout="wide")

st.title("🧪 CorroPredict AI")
st.subheader("AI-Based Prediction of Mild-Steel Weight Loss")
st.write(
    "Estimate mild-steel weight loss under aqueous corrosion conditions "
    "using NaCl concentration and exposure duration."
)

@st.cache_data
def load_data():
    file_path = Path(__file__).parent / "data" / "aqueous_corrosion_data.csv"
    return pd.read_csv(file_path)

@st.cache_resource
def train_model(data):
    X = data[["nacl_pct", "exposure_days"]]
    y = data["weight_loss_mg"]
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)
    return model

data = load_data()
X = data[["nacl_pct", "exposure_days"]]
y = data["weight_loss_mg"]
model = train_model(data)

st.info(
    "Scope: The prediction model uses the aqueous corrosion dataset only. "
    "It predicts weight loss from NaCl concentration and exposure duration. "
    "Temperature, pH and humidity are discussed separately in the theoretical study."
)

left, right = st.columns([1, 1])
with left:
    st.markdown("### Enter aqueous conditions")
    with st.form("prediction_form"):
        nacl = st.number_input(
            "NaCl concentration (%)",
            min_value=float(data["nacl_pct"].min()),
            max_value=float(data["nacl_pct"].max()),
            value=3.0,
            step=0.5
        )
        days = st.number_input(
            "Exposure duration (days)",
            min_value=int(data["exposure_days"].min()),
            max_value=int(data["exposure_days"].max()),
            value=30,
            step=1
        )
        submitted = st.form_submit_button("Predict weight loss", use_container_width=True)

    if submitted:
        prediction = model.predict(pd.DataFrame(
            {"nacl_pct": [nacl], "exposure_days": [days]}
        ))[0]
        st.success(f"Estimated weight loss: **{prediction:.3f} mg**")
        st.caption(
            "This is an estimate from the trained model, not an experimental measurement. "
            "Use inputs within the dataset ranges."
        )

with right:
    st.markdown("### Dataset overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Data points", len(data))
    m2.metric("NaCl range", f'{data["nacl_pct"].min():g}–{data["nacl_pct"].max():g}%')
    m3.metric("Duration range", f'{data["exposure_days"].min()}–{data["exposure_days"].max()} days')
    st.write("The model was trained using the aqueous corrosion data shown below.")

st.divider()
st.markdown("### Model evaluation")
with st.spinner("Calculating leave-one-out cross-validation results..."):
    loo = LeaveOneOut()
    cv_predictions = cross_val_predict(model, X, y, cv=loo, n_jobs=None)
    mae = mean_absolute_error(y, cv_predictions)
    rmse = np.sqrt(mean_squared_error(y, cv_predictions))
    r2 = r2_score(y, cv_predictions)

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.4f} mg")
c2.metric("RMSE", f"{rmse:.4f} mg")
c3.metric("R²", f"{r2:.4f}")
st.caption(
    "Metrics are calculated using leave-one-out cross-validation on this small dataset. "
    "They describe performance on these data and do not guarantee accuracy for other conditions."
)

st.divider()
st.markdown("### Data visualisation")
tab1, tab2, tab3 = st.tabs(["NaCl concentration", "Exposure duration", "Actual vs predicted"])

with tab1:
    grouped = data.groupby("nacl_pct", as_index=False)["weight_loss_mg"].mean()
    fig, ax = plt.subplots()
    ax.plot(grouped["nacl_pct"], grouped["weight_loss_mg"], marker="o")
    ax.set_xlabel("NaCl concentration (%)")
    ax.set_ylabel("Average weight loss (mg)")
    ax.set_title("Effect of NaCl concentration on weight loss")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    grouped = data.groupby("exposure_days", as_index=False)["weight_loss_mg"].mean()
    fig, ax = plt.subplots()
    ax.plot(grouped["exposure_days"], grouped["weight_loss_mg"], marker="o")
    ax.set_xlabel("Exposure duration (days)")
    ax.set_ylabel("Average weight loss (mg)")
    ax.set_title("Effect of exposure duration on weight loss")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    fig, ax = plt.subplots()
    ax.scatter(y, cv_predictions)
    low = min(float(y.min()), float(cv_predictions.min()))
    high = max(float(y.max()), float(cv_predictions.max()))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual weight loss (mg)")
    ax.set_ylabel("Cross-validated predicted weight loss (mg)")
    ax.set_title("Actual vs cross-validated predicted values")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.markdown("### Feature importance")
importance = pd.DataFrame({
    "Parameter": ["NaCl concentration", "Exposure duration"],
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)
fig, ax = plt.subplots()
ax.barh(importance["Parameter"], importance["Importance"])
ax.set_xlabel("Relative importance")
ax.set_title("Random Forest feature importance")
st.pyplot(fig)
plt.close(fig)

with st.expander("View dataset"):
    st.dataframe(data, use_container_width=True)
    st.download_button(
        "Download dataset (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="aqueous_corrosion_data.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption(
    "Dataset reference: Ali et al. (2020), A Simple Empirical Model for Predicting "
    "Weight Loss of Mild Steel due to Corrosion in NaCl Solution, International Journal "
    "of Automotive and Mechanical Engineering, 17(1), 7784–7791."
)
