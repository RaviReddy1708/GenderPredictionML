import argparse, pandas as pd

POS = set("love awesome great good happy excited enjoy like amazing".split())
NEG = set("hate bad terrible angry sad upset".split())

def sentiment_score(text):
    toks = text.split()
    pos = sum(1 for t in toks if t in POS)
    neg = sum(1 for t in toks if t in NEG)
    return (pos - neg) / max(1, len(toks))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.inp)
    col = "clean_text" if "clean_text" in df.columns else "text"
    df["sentiment_score"] = df[col].map(sentiment_score)
    df.to_csv(args.out, index=False)
    print(f"Added sentiment -> {args.out}")

if __name__ == "__main__":
    main()