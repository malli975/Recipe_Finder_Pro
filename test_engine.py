import pandas as pd
import pickle

# Load the data and the trained pieces
df = pd.read_csv('recipes.csv')
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def get_recommendations(my_ingredients):
    # Convert your ingredients to numbers
    query_vec = vectorizer.transform([my_ingredients])
    
    # Find the top 3 closest matches
    distances, indices = model.kneighbors(query_vec, n_neighbors=3)
    
    print(f"\nResults for: {my_ingredients}")
    for idx in indices[0]:
        print(f"Recipe: {df.iloc[idx]['name']}")
        print(f"Ingredients needed: {df.iloc[idx]['ingredients']}\n")

# Try it out!
get_recommendations("eggs flour sugar")