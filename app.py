import streamlit as st
import pickle
import re
import nltk

nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# LOAD MODEL & VECTORIZER
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# TEXT CLEANING
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words and len(w) > 2]

    return " ".join(words)

# PAGE CONFIG
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("💬 Twitter Sentiment Analyzer")

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]

        if result == 1:
            st.success("Positive Sentiment 😊")   # ✅ FIXED (logic corrected)
        else:
            st.error("Negative Sentiment 😡")