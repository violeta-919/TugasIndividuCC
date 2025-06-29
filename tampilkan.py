# Tambahkan ini di atas
import streamlit as st

# Tampilkan judul
st.title("Analisis Review")

# Input dari user
user_input = st.text_area("Masukkan Review Film Anda:")

# Proses hanya jika ada input
if user_input:
    processed = preprocess_text(user_input)
    vectorized = vectorizer.transform([processed])
    prob = model.predict_proba(vectorized)[0]
    prediction = labels[np.argmax(prob)]
    
    st.markdown("### Hasil Prediksi:")
    st.write(f"Review ini kemungkinan **{prediction.upper()}**")
    st.write(f"Probabilitas: Negative = {prob[0]:.2f}, Positive = {prob[1]:.2f}")
