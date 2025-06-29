import streamlit as st
import joblib
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download resource NLTK yang dibutuhkan
nltk.download('punkt')
nltk.download('stopwords')

# Load model dan vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function (harus sama dengan yang dipakai saat training)
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# UI Streamlit
st.set_page_config(page_title="Analisis Sentimen", layout="centered")
st.title("🎬 Analisis Sentimen Ulasan Film IMDB")
st.write("Masukkan ulasan film, lalu klik tombol **Analisis** untuk memprediksi apakah ulasan tersebut positif atau negatif.")

# Input user
review_input = st.text_area("Tulis ulasan film Anda di sini", height=200)

if st.button("Analisis"):
    if review_input.strip() == "":
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        cleaned = preprocess_text(review_input)
        vectorized = vectorizer.transform([cleaned])
        probabilities = model.predict_proba(vectorized)[0]
        label = ['negative', 'positive']
        prediction = label[np.argmax(probabilities)]

        st.markdown("### Hasil Analisis:")
        st.success(f"**Sentimen:** {prediction.capitalize()}")
        st.write("**Probabilitas:**")
        st.write({
            "Negatif": f"{probabilities[0]*100:.2f}%",
            "Positif": f"{probabilities[1]*100:.2f}%"
        })
