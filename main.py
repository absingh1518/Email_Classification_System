import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import string
import re
# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')


# Initialize tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_email(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and stem
    processed_tokens = [stemmer.stem(token) for token in tokens 
                       if token not in stop_words and token not in string.punctuation]
    
    return processed_tokens

# Load and prepare training data
def prepare_training_data(emails, labels):
    # emails: list of email texts
    # labels: list of corresponding labels
    
    X = []  # Features
    y = labels  # Target labels
    
    # Load GloVe embeddings
    embeddings_index = {}
    with open('glove.6B.100d.txt', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Convert each email to vector
    for email in emails:
        processed_tokens = preprocess_email(email)
        email_vector = np.zeros(100)  # GloVe dimension
        word_count = 0
        
        for token in processed_tokens:
            if token in embeddings_index:
                email_vector += embeddings_index[token]
                word_count += 1
        
        if word_count > 0:
            email_vector = email_vector / word_count  # Average of word vectors
        
        X.append(email_vector)
    
    return np.array(X), np.array(y)

# Example usage with sample data
sample_emails = [
    "Meeting scheduled for tomorrow at 2 PM",
    "Customer complaint about product quality",
    "Please process this transaction",
    "Maintenance required for server room"
]

sample_labels = [
    "scheduling",
    "complaint",
    "transaction",
    "maintenance"
]

# Prepare data
X, y = prepare_training_data(sample_emails, sample_labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM classifier
svm_classifier = SVC(C=100, gamma=0.01, kernel='rbf')
svm_classifier.fit(X_train, y_train)

# Function to classify new email
def classify_email(email_text):
    # Preprocess and vectorize new email
    processed_tokens = preprocess_email(email_text)
    email_vector = np.zeros(100)
    word_count = 0
    
    for token in processed_tokens:
        if token in embeddings_index:
            email_vector += embeddings_index[token]
            word_count += 1
    
    if word_count > 0:
        email_vector = email_vector / word_count
    
    # Reshape for prediction
    email_vector = email_vector.reshape(1, -1)
    
    # Predict category
    prediction = svm_classifier.predict(email_vector)
    return prediction[0]


predicted_category = classify_email(new_email)
print(f"Predicted category: {predicted_category}")
