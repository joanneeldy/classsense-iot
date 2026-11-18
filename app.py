"""ClassSense: classroom conditions and anomaly detection demo."""

import pandas as pd
import streamlit as st

from core import FEATURES, build_experiment, demo_recommendations, rule_predict

st.set_page_config(page_title="ClassSense | Classroom Monitoring", page_icon="🏫", layout="wide")


@st.cache_resource

def load_experiment():
    return build_experiment()


model, train, test, metrics = load_experiment()

st.title("ClassSense")
st.subheader("Classroom environmental monitoring")
st.caption("Exploratory IoT + machine learning prototype | Simulated readings only")

with st.expander("About this prototype", expanded=False):
    st.write(
        "This dashboard explores environmental monitoring for a classroom with "
        "two air conditioners, two ceiling fans and three lighting switch groups. "
        "The controls below simulate sensor readings; no physical devices are connected."
    )
    st.write(
        "A threshold rule checks specified limits, while Isolation Forest "
        "flags observations unusual relative to synthetic training data. "
        "Neither approach provides a validated safety or comfort assessment."
    )

st.header("Live scenario simulator")
st.write("Adjust the readings to compare threshold alerts with ML anomaly detection.")
left, middle, right = st.columns(3)
with left:
    temperature = st.slider("Temperature (°C)", 20.0, 38.0, 28.0, 0.5)
with middle:
    humidity = st.slider("Humidity (%)", 30.0, 95.0, 63.0, 1.0)
with right:
    light = st.slider("Illuminance (lux)", 50, 900, 420, 10)

reading = pd.DataFrame([{
    "temperature_c": temperature,
    "humidity_pct": humidity,
    "light_lux": light,
}])[FEATURES]
rule_alert = bool(rule_predict(reading)[0])
ml_alert = bool(model.predict(reading)[0] == -1)
anomaly_score = -float(model.decision_function(reading)[0])

rule_col, ml_col, score_col = st.columns(3)
rule_col.metric("Threshold rules", "Alert" if rule_alert else "Normal")
ml_col.metric("Isolation Forest", "Anomaly" if ml_alert else "Normal")
score_col.metric("Anomaly score", f"{anomaly_score:.3f}")
st.caption("Higher scores mean more unusual readings under this trained model; this is not a risk probability.")

if rule_alert or ml_alert:
    st.warning("Review recommended: one or both methods flagged this reading.")
else:
    st.success("No alerts for this simulated reading.")

st.markdown("**Suggested checks**")
for recommendation in demo_recommendations(temperature, humidity, light):
    st.write(f"• {recommendation}")

st.divider()
st.header("Model evaluation")
st.write(
    "The comparison uses 700 assumed-normal synthetic training samples and "
    "300 held-out test samples (240 normal, 60 deliberately injected anomalies)."
)

shown_metrics = metrics.copy()
for field in ["precision", "recall", "f1", "false_positive_rate"]:
    shown_metrics[field] = shown_metrics[field].map(lambda x: f"{x:.3f}")
st.dataframe(shown_metrics, hide_index=True, use_container_width=True)
st.caption("Performance is against synthetic labels, not observed classroom events.")

st.subheader("Example test readings")
st.line_chart(test[FEATURES].head(80), use_container_width=True)

with st.expander("Explore labeled test cases"):
    result_table = test.copy()
    result_table["ground_truth"] = result_table["ground_truth"].map({0: "Normal", 1: "Injected anomaly"})
    result_table["rule_alert"] = result_table["rule_alert"].map({0: "Normal", 1: "Alert"})
    result_table["ml_alert"] = result_table["ml_alert"].map({0: "Normal", 1: "Anomaly"})
    st.dataframe(result_table, hide_index=True, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "Download evaluation results",
        data=metrics.to_csv(index=False),
        file_name="evaluation_metrics.csv", mime="text/csv", use_container_width=True,
    )
with col_b:
    st.download_button(
        "Download synthetic test data",
        data=test.to_csv(index=False),
        file_name="synthetic_test_data.csv", mime="text/csv", use_container_width=True,
    )

st.divider()
st.caption(
    "Scope: software demonstration only. The proposed ESP32/Wokwi sensor setup is "
    "not connected to this app, and no air conditioners, fans or lights are controlled."
)
