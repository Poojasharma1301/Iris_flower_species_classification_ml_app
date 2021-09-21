from fastapi import FastAPI
from typing import Optional
import uvicorn
import numpy as np
import pandas as pd
import pickle
app1 = FastAPI(title="Iris Classification",
    description="A simple API tp predict the irsi classification ",
    version="0.1",)
from pydantic import BaseModel
class Text(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float    
model = pickle.load(open('dtc.pkl',"rb"))    
@app1.get("/greet/{text}")
def greeting(text:str):
    return {"Hi {} welcome to Iris Flower Classification ML App".format(text)}


@app1.post("/classify")
def classify(item:Text):
    input_value=item.dict()
    data=pd.DataFrame([input_value])
    prediction=model.predict(data)
    output = int(prediction[0])
    # output dictionary
    sentiments = {0: "Iris-setosa", 1: "Iris-versicolor",2: "Iris-virginica"}

    # show results
    result = {sentiments[output]}
    return result 