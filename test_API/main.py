#shell : uvicorn main:app --reload

from fastapi import FastAPI
import pandas as pd
import json

app = FastAPI()

df = json.loads(pd.DataFrame({
    'films_1':['Il faut sauver Ryan Burcke','Platoon','Rambo II'],
    'films_2':['Freddy IV','Gremlins II','Bienvenue à Zombieland'],
    'films_3':['Princess Mononoke','Cobra - Space Adventures','Hunter x Hunter']
    }).to_json())

  
@app.get('/line/')
async def root():
    return df

@app.post('/add/')
async def add_line(data2):
    return data2

@app.post('/choice/')
async def add_line(data_1, data_2, data_3):
    return [data_1, data_2, data_3]

