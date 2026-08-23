from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


texts = [
    "I want a refund for my order",
    "How can I get my money back",
    "The product was damaged and I need a refund",
    "Please help me return this item",
    "I have not received my refund yet",
    "Can I cancel my purchase and get a refund",
    "The refund process is taking too long",
    "I would like to return this product",

    "My order arrived late",
    "The delivery was delayed",
    "Where is my package",
    "The tracking information has not updated",
    "When will my order arrive",
    "The courier missed the delivery",
    "My package is still in transit",
    "I have a question about delivery",
]

labels = [
    "refund",
    "refund",
    "refund",
    "refund",
    "refund",
    "refund",
    "refund",
    "refund",

    "delivery",
    "delivery",
    "delivery",
    "delivery",
    "delivery",
    "delivery",
    "delivery",
    "delivery",
]


X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels,
)


model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000)),
])


model.fit(X_train, y_train)

predictions = model.predict(X_test)


print("Test examples:")
for text, actual, predicted in zip(X_test, y_test, predictions):
    print(f"Text: {text}")
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")
    print()


print("Evaluation:")
print("Accuracy:", accuracy_score(y_test, predictions))
print(
    "Precision:",
    precision_score(y_test, predictions, average="weighted", zero_division=0),
)
print(
    "Recall:",
    recall_score(y_test, predictions, average="weighted", zero_division=0),
)
print(
    "F1:",
    f1_score(y_test, predictions, average="weighted", zero_division=0),
)


new_texts = [
    "I need my money returned",
    "My parcel has not arrived",
]

new_predictions = model.predict(new_texts)

print("New predictions:")
for text, prediction in zip(new_texts, new_predictions):
    print(f"{text} -> {prediction}")