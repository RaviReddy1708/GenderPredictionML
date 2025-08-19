import argparse, pandas as pd, joblib, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.inp)
    X = df[["clean_text", "sentiment_score"]]
    y = df["gender_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    features = ColumnTransformer([("tfidf", TfidfVectorizer(max_features=1000), "clean_text"),
                                  ("sent", "passthrough", ["sentiment_score"])])
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "nb": MultinomialNB(),
        "knn": KNeighborsClassifier(n_neighbors=3),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": LinearSVC()
    }
    report = ["# Accuracy Report", ""]
    model_dir = Path("models"); model_dir.mkdir(exist_ok=True)
    for name, clf in models.items():
        pipe = Pipeline([("feat", features if name!="nb" else ColumnTransformer([("tfidf", TfidfVectorizer(max_features=1000), "clean_text")])), ("clf", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report.append(f"- {name}: {acc:.3f}")
        joblib.dump(pipe, model_dir/f"{name}.joblib")
        cm = confusion_matrix(y_test, preds, labels=sorted(y.unique()))
        plt.imshow(cm, cmap="Blues"); plt.title(name); plt.savefig(model_dir/f"cm_{name}.png"); plt.close()
    Path(args.report).write_text("\n".join(report))
    print("Training done. Report saved.")

if __name__ == "__main__":
    main()