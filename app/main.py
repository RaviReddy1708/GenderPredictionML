import streamlit as st, pandas as pd, joblib

st.title("Gender Prediction Demo (College Project)")

txt = st.text_area("Enter text:", "I love playing football and coding.")
model_choice = st.selectbox("Choose Model", ["logreg","rf","svm"])

if st.button("Predict"):
    try:
        model = joblib.load(f"models/{model_choice}.joblib")
        df = pd.DataFrame([{"clean_text": txt, "sentiment_score": 0.0}])
        pred = model.predict(df)[0]
        st.success(f"Predicted Gender: {pred}")
    except Exception as e:
        st.error(f"Error: {e}. Train models first.")