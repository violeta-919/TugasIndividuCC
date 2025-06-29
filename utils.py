import nltk
nltk.data.path.append('./nltk_data')  # arahkan ke folder lokal
nltk.download('punkt', download_dir='./nltk_data', quiet=True)
nltk.download('stopwords', download_dir='./nltk_data', quiet=True)

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Membersihkan dan memproses teks untuk digunakan pada model klasifikasi.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)