import streamlit as st
import pandas as pd, numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile

import config
import model_pipeline as mp

st.set_page_config(page_title="Parkinsons – ML App (Pro, v3)", layout="wide")
st.title("🧪 Parkinsons – ML App (Pro, v3)")

# Sidebar – model & params
st.sidebar.header("Model & Params")
model_options = ["LogisticRegression","RandomForest","SVC","XGBoost","MLP","KerasNN"]
model_name = st.sidebar.selectbox("Model", model_options, index=model_options.index(config.DEFAULT_MODEL))
params = config.DEFAULT_PARAMS.get(model_name, {}).copy()

def _edit_param(key, val):
    if isinstance(val, bool): return st.sidebar.checkbox(key, value=val)
    if isinstance(val, int): return st.sidebar.number_input(key, value=int(val), step=1)
    if isinstance(val, float): return st.sidebar.number_input(key, value=float(val))
    return val

for k,v in params.items():
    params[k] = _edit_param(k, v)

st.sidebar.markdown("---")
do_cv = st.sidebar.checkbox("Run Cross-Validation", value=True)
do_tune = st.sidebar.checkbox("Run GridSearch Tuning (top model)", value=True)

# Load data (auto-creates if missing)
st.subheader("1) Data")
df = mp.load_data(config.TRAIN_DATA_PATH)
st.dataframe(df.head())

# EDA
def eda_section(df: pd.DataFrame):
    st.subheader("0) EDA")
    features = config.FEATURES; target = config.TARGET
    st.write("**Shape:**", df.shape)
    st.write("**Missing values (top 20):**")
    st.dataframe(df[features + [target]].isna().sum().sort_values(ascending=False).head(20))
    st.write("**Descriptive stats:**")
    st.dataframe(df[features].describe().T)

    # Class balance
    st.write("**Class balance:**")
    cls = df[target].value_counts().rename({0:"No-PD", 1:"PD"})
    st.bar_chart(cls)

    # Correlation heatmap
    st.write("**Correlation heatmap:**")
    corr = df[features + [target]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, ax=ax, cmap="vlag", center=0)
    st.pyplot(fig)

    # Key feature distributions
    key_feats = [f for f in ["MDVP:Fo(Hz)","MDVP:Jitter(%)","MDVP:Shimmer","HNR","RPDE","DFA","PPE"] if f in features]
    st.write("**Feature distributions by class (selected):**")
    for f in key_feats:
        fig, ax = plt.subplots(figsize=(5,3))
        sns.kdeplot(data=df, x=f, hue=target, common_norm=False, fill=True, alpha=0.3, ax=ax)
        ax.set_title(f); st.pyplot(fig)

with st.expander("🔎 Run EDA (click to expand)", expanded=False):
    eda_section(df)

# Train
st.subheader("2) Train")
col1, col2 = st.columns(2)
with col1:
    if st.button("Train candidate model"):
        res = mp.train_model(config.TRAIN_DATA_PATH, model_name=model_name, model_params=params, do_cv=do_cv, do_tune=do_tune)
        if not res.get("ok", False):
            st.error("\n".join(res.get("errors", [])))
        else:
            st.success(f"Candidate saved to {res['candidate_path']}")
            st.json(res["val_metrics"])
            if res.get("cv_means"): st.write("CV means:"); st.json(res["cv_means"])
with col2:
    st.write("Artifacts:")
    for p,cap in [("assets/roc.png","ROC"),("assets/pr.png","PR"),("assets/cm.png","Confusion Matrix")]:
        if Path(p).exists(): st.image(p, caption=cap)

# Evaluate
st.subheader("3) Evaluate (load & test saved model)")
eval_target = st.selectbox("Evaluate which model?", ["Candidate", "Production"])
if st.button("Evaluate now"):
    path = config.TEMP_MODEL_PATH if eval_target=="Candidate" else config.MODEL_PATH
    try:
        mets = mp.evaluate_model(path)
        st.json(mets)
        for p,cap in [("assets/roc.png","ROC"),("assets/pr.png","PR"),("assets/cm.png","Confusion Matrix")]:
            if Path(p).exists(): st.image(p, caption=cap)
    except Exception as e:
        st.error(str(e))

# Promote
st.subheader("4) Promote candidate → production")
if st.button("Promote!"):
    try:
        msg = mp.promote_model_to_production()
        st.success(msg)
    except Exception as e:
        st.error(str(e))

# Predict single
st.subheader("5) Predict – single row")
with st.expander("Upload single-row CSV with the exact feature headers"):
    st.caption("Use assets/single_row_template.csv as a template (no 'name'/'status').")
    up_single = st.file_uploader("Upload single-row CSV", type=["csv"], key="single")
    if st.button("Run single prediction"):
        try:
            if up_single is None:
                st.error("Please upload a one-row CSV.")
            else:
                df_in = pd.read_csv(up_single)
                pred, proba = mp.run_prediction(df_in.iloc[:1])
                st.success(f"Prediction: {pred} | PD probability: {proba:.3f}")
        except Exception as e:
            st.error(str(e))

# Batch predict
st.subheader("6) Batch predict – CSV")
up = st.file_uploader("Upload CSV with feature columns only (no 'name'/'status')", type=["csv"], key="batch")
if st.button("Run batch predictions"):
    if up is None:
        st.error("Please upload a CSV.")
    else:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(up.read()); tmp_path = tmp.name
            out = mp.batch_predict(tmp_path)
            st.dataframe(out.head())
            st.download_button("Download predictions CSV", data=out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(str(e))

st.markdown("---")
st.caption("KerasNN requires scikeras + tensorflow-cpu; otherwise choose XGBoost/LogReg/etc.")
