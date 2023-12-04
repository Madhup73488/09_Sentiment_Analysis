import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report

# Reading data from extracted_data.csv file
data = pd.read_csv('extracted_data.csv')

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['clean_tweet'],\
                data['sentiment_value'], test_size=0.2, random_state=42)

# Text preprocessing: Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize and train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

# Input a keyword from the user
user_input = input("Enter a keyword: ")

# Transform the user input into a numerical feature using the trained CountVectorizer
user_input_vectorized = vectorizer.transform([user_input])

# Predict the sentiment for the user input
predicted_sentiment = nb_classifier.predict(user_input_vectorized)

print(f"The predicted sentiment for '{user_input}' is: {predicted_sentiment[0]}")
# Predict on the test set
predictions = nb_classifier.predict(X_test_vectorized)


# Evaluate the model
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy: {accuracy:.2f}")

# Print classification report
# print(classification_report(y_test, predictions))
