# SMS Spam Classification Model (NLP)

A High-Performance Natural Language Processing (NLP) model engineered to classify SMS messages as either **Spam** or **Ham** (Legitimate). 

This model achieves an exceptional **Accuracy Score of 98.47%**, making it reliable for real-world text filtering applications.

## 🚀 Performance Metrics
* **Accuracy:** 98.47%
* **Domain:** Natural Language Processing (NLP)
* **Task:** Binary Text Classification

## 🛠️ Features
* **Text Preprocessing:** Tokenization, stop-word removal, lemmatization, and case normalization.
* **Vectorization:** Text numerical representation using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization / CountVectorizer.
* **Classifier:** Machine Learning / Deep Learning architecture optimized for sparse text structures.

## 📁 Repository Structure
```text
├── dataset/            # Contains the SMS Spam Collection dataset
├── notebooks/          # Jupyter Notebooks detailing EDA, training, and evaluation
├── models/             # Saved model binaries and vectorizers (.pkl, .h5, etc.)
├── src/                # Source code files for deployment
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd sms-spam-classification
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage & Inference

You can use the trained model pipeline to test your own text messages. Run the python script or execution flow as follows:

```python
import joblib

# Load the saved model and vectorizer
model = joblib.load('models/spam_classifier.pkl')
tfidf = joblib.load('models/vectorizer.pkl')

def predict_message(text):
    # Preprocess and vectorize input
    transformed_text = tfidf.transform([text])
    # Predict
    prediction = model.predict(transformed_text)
    return "Spam" if prediction[0] == 1 else "Ham"

# Example usage
sample_text = "URGENT! You have won a 1-week FREE membership to our prize club! Text 'CLAIM' to 81010."
print(f"Result: {predict_message(sample_text)}")
```

## 📊 Dataset
The model was trained and evaluated on the public **SMS Spam Collection Dataset**, which consists of a set of SMS tagged messages that have been collected for SMS Spam research.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
