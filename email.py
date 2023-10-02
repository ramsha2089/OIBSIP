import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the CSV file
data = pd.read_csv(r"C:\project\ramsha\ramsha\New folder/spam.csv", encoding="latin1")

# Drop unwanted columns
col = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]
data = data.drop(columns=col)

# Replace "ham" and "spam" with 0 and 1 in the "v1" column
data.replace({"v1": {"ham": 0, "spam": 1}}, inplace=True)

# Define features (X) and target (Y)
X = data["v2"]
Y = data["v1"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create an instance of the Logistic Regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train_tfidf, Y_train)

# Make predictions on the test data
Y_pred = model.predict(X_test_tfidf)

# Evaluate the model on the test data
accuracy = model.score(X_test_tfidf, Y_test)
print("Accuracy:", accuracy)

# Now, let's predict the label for a new email
new_email = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
# Vectorize the new email using the same TF-IDF vectorizer
new_email_tfidf = tfidf_vectorizer.transform(new_email)
# Make predictions for the new email
email_predict = model.predict(new_email_tfidf)
print("Email predict:", email_predict)
