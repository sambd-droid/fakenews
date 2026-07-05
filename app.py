
import streamlit as st
import joblib
import re
import string

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_files():
    model = joblib.load("fake_news_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_files()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("📰 Fake News Detection System")
st.write("Enter a news headline or full news article to classify it as Fake or Real.")

news_text = st.text_area(
    "Enter News Text",
    placeholder="Paste your news headline or article here...",
    height=220
)

if st.button("Predict"):

    if not news_text.strip():
        st.warning("Please enter news text first.")

    else:
        cleaned_news = clean_text(news_text)
        news_vector = vectorizer.transform([cleaned_news])

        prediction = model.predict(news_vector)[0]
        probabilities = model.predict_proba(news_vector)[0]
        confidence = max(probabilities) * 100

        if prediction == 0:
            st.error("Prediction: Fake News")
        else:
            st.success("Prediction: Real News")

        st.info(f"Prediction confidence: {confidence:.2f}%")

st.divider()

st.caption(
    "This system gives a machine-learning prediction based on training data. "
    "It should not replace professional fact-checking."
)
