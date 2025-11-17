import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = Path("labeled")

df = pd.concat((pd.read_csv(f) for f in DATA_DIR.glob("*.csv")), ignore_index=True)
print(len(df))
X = df["description"].astype(str)
y = df["label"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 3)),
    MultinomialNB()
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
labels = sorted(df["label"].unique())

cm = confusion_matrix(y_test, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap="Blues", colorbar=True)


print("Test accuracy:", model.score(X_test, y_test))

example = ["SQ *JET RAG 825 North La Brea LOS ANGELES 90038 CA USA 2%"]
print("Prediction:", model.predict(example)[0])


plt.xticks(rotation=45, ha="right")
plt.title("Naive Bayes Confusion Matrix")
plt.tight_layout()
plt.show()