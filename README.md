
# 📰 Fake News Detection Using Machine Learning

This project aims to detect whether a news article is **Fake** or **Real** based on its text content. It uses Natural Language Processing (NLP) and a machine learning model trained on labeled news datasets.

Built with 🧠 Scikit-learn + 🧪 TF-IDF + 📊 Streamlit + 🔍 LIME (Explainability).

---

## 🚀 Features

- ✅ Predict whether news text is **Fake** or **Real**
- 🧠 ML model trained with TF-IDF + Logistic Regression
- 🔍 LIME Explainability shows which words influenced the prediction
- 🌐 Streamlit-based web interface for easy interaction

---

## 🖥️ Demo

Paste news text into the input box and see whether it's classified as Fake or Real, along with a color-coded explanation.

![Demo Screenshot](demo.png) <!-- Replace with your own screenshot if available -->

---

## 📂 Project Structure

```
├── app.py                  # Streamlit web app
├── fake_news_model.pkl     # Trained ML model (not included in repo if >100MB)
├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files like large CSVs
└── README.md               # Project overview
```

---

## ⚙️ How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 📚 Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- TF-IDF (Text Vectorization)
- LIME (Explainability)
- Streamlit (Web App)

---

## 🧠 Future Improvements

- Add multi-class classification (e.g., satire, opinion, propaganda)
- Use deep learning (BERT) for better context understanding
- Host the app using Streamlit Cloud

---

## 📜 License

MIT License © Chinmayee S. Bharadwaj

---

## 🙌 Acknowledgements

- Dataset adapted from open-source Kaggle datasets
- Inspired by real-world fake news detection challenges
