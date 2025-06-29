import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import learning_curve

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

file_path = "C:/Users/wioll/Downloads/IMDB-Dataset.csv"

try:
    df = pd.read_csv(file_path)
    print("--- Data Berhasil Dimuat dari Excel ---")
    print("5 Baris pertama dari data anda:")
    print(df.head())
    print("\n" + "="*50 + "\n")
except FileNotFoundError:
    print(f"Error: File tidak ditemukan di {file_path}")
    print("Pastikan nama file dan path sudah benar.")
    exit()

# Ambil kolom teks dan label
x = df['review']     # Ganti jika nama kolom berbeda
y = df['sentiment']

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]','', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['review_bersih'] = df['review'].apply(preprocess_text)

print("--- Data Setelah Preprocessing ---")
print(df[['review_bersih', 'sentiment']].head())
print("\n" + "="*50 + "\n")

X = df['review_bersih']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_vectorized, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', label="Training Accuracy")
plt.plot(train_sizes, test_mean, 'o-', label="Validation Accuracy")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve for Existing Naive Bayes Model")
plt.legend()
plt.grid(True)
plt.show()

model.fit(X_train_vectorized, y_train)

print("--- Model Berhasil Dilatih ---")
print(f"Jumlah data latih: {X_train_vectorized.shape[0]}")
print(f"Jumlah data uji: {X_test_vectorized.shape[0]}")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy * 100:.2f}%\n")

print("--- Klasifikasi ---")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
class_labels = sorted(y.unique())

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()

review_positif = "Great movie! The acting was solid, and the plot kept me engaged the whole time."
review_negatif = "What a waste of time. The plot was predictable and the characters were incredibly boring."
reviews = [review_positif, review_negatif]

# Label asli (hanya untuk perbandingan kita)
sentimen_asli = ['positive', 'negative']

processed_reviews = [preprocess_text(rev) for rev in reviews]

X_pred_vec = vectorizer.transform(processed_reviews)

pred = model.predict_proba(X_pred_vec)

labels = ['negative', 'positive']
for i in range(len(reviews)):
    print("\n==============================================")
    print("Ulasan:", reviews[i])
    print("Sentimen Asli:", sentimen_asli[i])
    # Ambil indeks dengan probabilitas tertinggi, lalu cocokkan dengan label
    print("Prediksi Model:", labels[np.argmax(pred[i])])
    print("Rincian Probabilitas (Neg, Pos):", pred[i])

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")