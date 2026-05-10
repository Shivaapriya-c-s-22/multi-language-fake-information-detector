import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Loading dataset...")

try:
    # Load the data
    df = pd.read_csv('news.csv')
    
    # --- SMART DATA CLEANING ---
    # 1. Fill empty text with empty string to prevent crashes
    df['text'] = df['text'].fillna('')
    
    # 2. Clean the labels: Make UPPERCASE and remove extra spaces
    # This ensures "fake", "Fake ", and "FAKE" are all treated the same
    df['label'] = df['label'].astype(str).str.upper().str.strip()

    # Check what we are working with
    print(f"Dataset loaded with {len(df)} rows.")
    print(df['label'].value_counts()) # This tells us how many Fake vs Real articles we have

    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.2, random_state=7)

except FileNotFoundError:
    print("Error: 'news.csv' not found.")
    exit()
except Exception as e:
    print(f"Error reading data: {e}")
    exit()

# 2. PROCESS THE DATA
# Initialize TfidfVectorizer with stop_words removed (the, a, an...)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
tfidf_test = tfidf_vectorizer.transform(X_test)

# 3. TRAIN THE BRAIN
# PassiveAggressiveClassifier is good for text
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# 4. CHECK ACCURACY
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'\nModel Accuracy: {round(score*100, 2)}%\n')

# 5. SAVE THE BRAIN
with open('model.pkl', 'wb') as f:
    pickle.dump(pac, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("Success! New 'model.pkl' and 'vectorizer.pkl' created.")