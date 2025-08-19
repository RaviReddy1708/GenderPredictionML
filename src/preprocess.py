import argparse, re, pandas as pd

STOPWORDS = set("a the and is are to of in for with on at from by this that it be or as an will just".split())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(toks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.inp)
    df["clean_text"] = df["text"].astype(str).map(clean_text)
    df.to_csv(args.out, index=False)
    print(f"Preprocessed -> {args.out}")

if __name__ == "__main__":
    main()