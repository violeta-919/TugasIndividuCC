from utils import preprocess_text
import streamlit as st
import joblib
import numpy as np

# Load model dan vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# UI Streamlit
st.set_page_config(page_title="Analisis Sentimen", layout="centered")
st.title("🎬 Analisis Sentimen Ulasan Film IMDB")
st.write("Masukkan ulasan film anda dan kami akan menentukan apakah ulasan tersebut positif atau negatif.")

# Input user
review_input = st.text_area("Tulis ulasan film Anda di kolom ini (dengan bahasa Inggris)!", height=200)

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
        st.success(f"**Sentimen ulasan menunjukkan bawah ulasan anda** {prediction.capitalize()}")
