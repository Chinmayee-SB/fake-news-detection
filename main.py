

import pandas as pd
import numpy as np
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')


fake_df = pd.read_csv(r"C:\Users\Bharathi\OneDrive\Desktop\Projects\Fake news detection\archive\Fake.csv")
true_df = pd.read_csv(r"C:\Users\Bharathi\OneDrive\Desktop\Projects\Fake news detection\archive\True.csv")

fake_df['label'] = 0  # Fake news
true_df['label'] = 1  # Real news

df = pd.concat([fake_df, true_df], ignore_index=True)
print("Data loaded. Shape:", df.shape)
# === Step 1: Data Preprocessing ===

df['text'] = df['title'] + " " + df['text']
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(clean_text)

df.to_csv("cleaned_fake_news.csv", index=False)
print("Text cleaning complete. Cleaned dataset saved.")
print(df[['text', 'cleaned_text', 'label']].sample(3))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# === Step 2: TF-IDF Vectorization ===

vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

print("TF-IDF vectorization complete. Shape:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model training complete.")

y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved to disk.")

