import pandas as pd
# Reduce file size to stay under GitHub and Render free limits
df = pd.read_csv('recipes.csv')
df.head(4000).to_csv('recipes.csv', index=False)
print("Success! recipes.csv is now small enough to upload.")