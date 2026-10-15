import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="LungCare AI", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_random_forest.joblib")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Place best_random_forest.joblib beside app.py.")
    st.stop()

model = load_model(MODEL_PATH)

if hasattr(model, "feature_names_in_"):
    features = list(model.feature_names_in_)
else:
    features = [
        "Gender", "Air Pollution", "Alcohol use", "Dust Allergy",
        "OccuPational Hazards", "Genetic Risk", "chronic Lung Disease",
        "Balanced Diet", "Obesity", "Smoking", "Passive Smoker",
        "Chest Pain", "Coughing of Blood", "Fatigue", "Weight Loss",
        "Shortness of Breath", "Wheezing", "Swallowing Difficulty",
        "Clubbing of Finger Nails", "Frequent Cold", "Dry Cough", "Snoring"
    ]
    st.warning("The model does not expose its feature names. Confirm that this fallback list matches your training columns and their order.")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #071426 0%, #10263c 55%, #102f3b 100%); color: #f1f5f9; }
.block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
.hero { padding: 30px; border-radius: 22px; background: linear-gradient(120deg, #123c50, #176b70); border: 1px solid #347b83; margin-bottom: 24px; }
.hero h1 { color: white; font-size: 38px; margin-bottom: 8px; }
.hero p { color: #d6f4f1; font-size: 16px; }
.result { padding: 24px; border-radius: 18px; background: linear-gradient(120deg, #123e4a, #174f4a); border: 1px solid #3c8981; text-align: center; }
.note { padding: 14px; border-radius: 12px; background: rgba(255,255,255,0.06); border-left: 4px solid #4dd0c8; color: #dbeafe; }
div.stButton > button { width: 100%; border-radius: 12px; min-height: 48px; font-weight: 700; background: #4dd0c8; color: #06202c; border: none; }
div.stButton > button:hover { background: #8be9e3; color: #06202c; }
[data-testid="stSidebar"] { background: #081827; }
h2, h3 { color: #a5f3eb !important; }
label, p, li { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🫁 LungCare AI")
    st.caption("Machine-learning demonstration")
    st.divider()
    st.markdown("### How to use")
    st.write("1. Enter the requested feature values.")
    st.write("2. Click Predict.")
    st.write("3. Review the model output.")
    st.divider()
    st.caption("Model: Random Forest")
    st.caption("File: best_random_forest.joblib")

st.markdown("""
<div class="hero">
    <h1>🫁 LungCare AI</h1>
    <p>Explore a machine-learning prediction based on the risk factors and symptoms entered below.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="note">
<strong>Important:</strong> This is an educational machine-learning demo, not a medical diagnostic tool.
Its predictions may be inaccurate. Consult a qualified healthcare professional about health concerns.
</div>
""", unsafe_allow_html=True)

st.write("")
st.markdown("## Patient Feature Details")
st.write("Enter values that match the encoding and value ranges used when training your model.")

values = {}
left_col, right_col = st.columns(2)
midpoint = (len(features) + 1) // 2

for i, feature in enumerate(features):
    target_col = left_col if i < midpoint else right_col
    with target_col:
        if feature.strip().lower() == "gender":
            values[feature] = st.selectbox("Gender", options=[1, 2],
                help="Use the numeric encoding from your training dataset.")
        elif feature.strip().lower() == "age":
            values[feature] = st.number_input(feature, min_value=1, max_value=120, value=30, step=1)
        else:
            values[feature] = st.slider(feature, min_value=0, max_value=10, value=5, step=1,
                help="Confirm that this range matches the values used during model training.")

st.write("")
predict_clicked = st.button("🔍 Predict Model Output")

if predict_clicked:
    try:
        input_df = pd.DataFrame([[values[f] for f in features]], columns=features)
        prediction = model.predict(input_df)[0]

        st.markdown("## Prediction Result")
        st.markdown(f"""
<div class="result">
    <h3>Model-predicted class</h3>
    <h1 style="color:white;">{prediction}</h1>
</div>
""", unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            st.write("")
            st.markdown("### Model output probabilities")
            probability_df = pd.DataFrame({
                "Class": [str(c) for c in model.classes_],
                "Probability (%)": [round(float(p) * 100, 2) for p in probabilities]
            })
            st.dataframe(probability_df, use_container_width=True, hide_index=True)
            st.caption("These are model output probabilities, not a person's probability of having cancer.")

        st.write("")
        st.info("This result reflects the trained model's output only. It does not confirm or rule out lung cancer.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Check that the model file, input feature names, feature order, data types, and value ranges match your training setup.")

st.divider()
st.caption("LungCare AI • Educational ML Project • Not for medical diagnosis")
