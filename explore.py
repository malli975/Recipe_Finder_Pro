import pandas as pd

# Load the file we just renamed
df = pd.read_csv('recipes.csv')

# Print the columns to see what we are working with
print("--- Column Names ---")
print(df.columns.tolist())

# Look at the first few rows of ingredients
print("\n--- Ingredient Preview ---")
print(df['ingredients'].head())