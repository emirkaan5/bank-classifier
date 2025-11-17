import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

DATA_DIR = Path("labeled")
df = pd.concat((pd.read_csv(f) for f in DATA_DIR.glob("*.csv")), ignore_index=True)

X = df["description"].astype(str)
y = df["label"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    MultinomialNB()
)

model.fit(X_train, y_train)
print("Test accuracy:", model.score(X_test, y_test))

example = ["SQ *JET RAG 825 North La Brea LOS ANGELES 90038 CA USA 2%"]
print("Prediction:", model.predict(example)[0])
