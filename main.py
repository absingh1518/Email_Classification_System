# Load GloVe embeddings

embeddings_index = {}
with open('glove.6B.100d.txt', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Preprocess email content
def preprocess_email(text):
    # Remove stopwords, punctuation
    # Perform stemming
    return processed_text

# Create email vectors
def vectorize_email(email_text):
    tokens = preprocess_email(email_text)
    email_vector = np.zeros(100) # GloVe dimension
    for token in tokens:
        if token in embeddings_index:
            email_vector += embeddings_index[token]
    return email_vector

# Train SVM classifier
svm_classifier = SVC(C=100, gamma=0.01, kernel='rbf')
svm_classifier.fit(X_train, y_train)
