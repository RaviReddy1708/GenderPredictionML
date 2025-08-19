import argparse, pandas as pd, joblib, json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.inp)
    X = df[["clean_text", "sentiment_score"]]
    y = df["gender_label"]
    metrics = {}
    for model in ["logreg","nb","knn","rf","svm"]:
        path = Path("models")/f"{model}.joblib"
        if not path.exists():
            continue
        clf = joblib.load(path)
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        p,r,f,_ = precision_recall_fscore_support(y, preds, average="weighted", zero_division=0)
        metrics[model] = {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f)}
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics -> {args.out}")

if __name__ == "__main__":
    main()