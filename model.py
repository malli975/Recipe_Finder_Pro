import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle

def train_model():
    print("Loading data...")
    df = pd.read_csv('recipes.csv')
    
    # Fill any missing values to avoid errors
    df['ingredients'] = df['ingredients'].fillna('')
    
    # Clean the ingredients: remove brackets, quotes, and commas
    print("Cleaning ingredients...")
    df['clean_ingredients'] = df['ingredients'].str.replace(r"[\[\]']", "", regex=True).str.replace(",", "")
    
    # Step 2: Vectorization (Math conversion)
    print("Training the brain (Vectorizing)...")
    vectorizer = TfidfVectorizer()
    recipe_matrix = vectorizer.fit_transform(df['clean_ingredients'])
    
    # Step 3: The Search Model
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(recipe_matrix)
    
    # Save the 'brain' so we don't have to train it every time
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("Success! Model trained and saved.")
    return df

if __name__ == "__main__":
    train_model()