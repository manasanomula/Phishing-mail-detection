# Phishing Email Detection Model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

emails = [
    "Congratulations! You won a free iPhone. Click here now",
    "Urgent! Your bank account is blocked. Verify immediately",
    "Meeting scheduled for tomorrow at 10 AM",
    "Please find the project report attached",
    "Win cash prize now by clicking this link",
    "Your Amazon order has been shipped",
    "Update your password immediately",
    "Lunch with friends this weekend"
]

labels = [
    "Phishing",
    "Phishing",
    "Safe",
    "Safe",
    "Phishing",
    "Safe",
    "Phishing",
    "Safe"
]


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)


X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)


model = MultinomialNB()
model.fit(X_train, y_train)


new_email = ["Your account is suspended. Click here to verify"]
new_email_vector = vectorizer.transform(new_email)

prediction = model.predict(new_email_vector)

print("Email:", new_email[0])
print("Prediction:", prediction[0])