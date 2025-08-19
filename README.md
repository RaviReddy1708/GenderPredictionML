# Gender Prediction from Social Media Text (College Project)

This project predicts **gender** of users based on social media posts by combining **text preprocessing**, **sentiment analysis**, and **machine learning models**.

## Project Workflow
1. **Data Preprocessing** — clean raw text (lowercasing, stopwords removal, etc.).
2. **Sentiment Analysis** — extract sentiment score (positive/negative).
3. **Feature Fusion** — combine TF-IDF text features with sentiment score.
4. **Model Training** — train Logistic Regression, Naive Bayes, kNN, Random Forest, SVM.
5. **Evaluation** — compare accuracy and visualize confusion matrices.

## Run Instructions
```bash
pip install -r requirements.txt

# Step 1: Preprocess & add sentiment
python src/preprocess.py --in data/sample.csv --out data/clean.csv
python src/sentiment.py  --in data/clean.csv --out data/with_sentiment.csv

# Step 2: Train models
python src/train_models.py --in data/with_sentiment.csv --report results/accuracy_report.md

# Step 3: Evaluate
python src/evaluate.py --in data/with_sentiment.csv --out results/metrics.json

# Step 4: (Optional) Launch Streamlit demo
streamlit run app/main.py
```

## Example Results
- Accuracy improves when adding sentiment features (~2% uplift).
- Random Forest and SVM perform better than baseline Logistic Regression.

## Future Work
- Use deep learning (LSTMs, Transformers) for better accuracy.
- Test on larger datasets.
- Deploy as a web service for real-time predictions.

---
**Created as a college ML project.**