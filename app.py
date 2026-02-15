from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import pickle
import uvicorn
import random #

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# This line is the secret to fixing the images
templates.env.globals.update(random=random.randint) 

print("Initializing Malli Gourmet Pro...")
df = pd.read_csv('recipes.csv')
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "recipes": None})

@app.get("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, ingredients: str, filter_type: str = "All", max_time: int = 120):
    query_vec = vectorizer.transform([ingredients])
    distances, indices = model.kneighbors(query_vec, n_neighbors=30)
    res = df.iloc[indices[0]].copy()
    
    # Advanced Filtering Logic
    res = res[res['minutes'] <= max_time]
    if filter_type == "Healthy":
        res = res[~res['ingredients'].str.contains('butter|sugar|oil|lard', case=False)]
    elif filter_type == "Fast Food Style":
        res = res[res['minutes'] <= 25]
    
    res['difficulty'] = res['steps'].apply(lambda x: "Easy" if len(str(x).split(',')) < 8 else "Expert")
    
    final_data = res.head(12).to_dict(orient="records")
    return templates.TemplateResponse("index.html", {
        "request": request, "recipes": final_data, "query": ingredients, 
        "filter_type": filter_type, "max_time": max_time
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)