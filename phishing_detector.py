
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# Small demonstration dataset
emails = [
    "Urgent verify your account at http://fake-login.example",
    "Your bank account is suspended click to restore",
    "Congratulations you won a prize claim it now",
    "Update your password immediately at this link",
    "Your payment failed enter card details now",
    "Security alert confirm your identity urgently",
    "You have won a lottery send your details",
    "Click here to unlock your blocked account",
    "Your package is held pay a fee now",
    "Verify your credentials to avoid account closure",
    "Team meeting scheduled for tomorrow",
    "Please find the project report attached",
    "Your leave request has been approved",
    "Let's discuss the assignment next week",
    "The class timetable has been updated",
    "Reminder about today's team meeting",
    "Your internship project has been reviewed",
    "Please submit the weekly progress report",
    "The workshop begins at ten tomorrow",
    "Thank you for attending the training session"
]

labels = [1] * 10 + [0] * 10

df = pd.DataFrame({
    "email": emails,
    "label": labels
})

X_train, X_test, y_train, y_test = train_test_split(
    df["email"],
    df["label"],
    test_size=0.3,
    random_state=42,
    stratify=df["label"]
)

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    MultinomialNB()
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model Accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nEnter an email to check:")
email = input("> ")

result = model.predict([email])[0]

if result == 1:
    print("Prediction: PHISHING")
else:
    print("Prediction: SAFE")

print("\nNote: This is a demonstration model.")
