import streamlit as st
import requests
import json
import pickle
import numpy as np
model=pickle.load(open('dtc.pkl','rb'))
from PIL import Image
image = Image.open('iri.jpg')
def main():
    html_temp = """
    <div style="background-color:DodgerBlue;padding:10px">
    <h2 style="color:white;text-align:center;">IRIS FLOWER SPECIES CLASSIFICATION ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.image(image,use_column_width=True, caption='Iris Flower Classification',width=400)
    sepal_length =st.slider('Enter Sepal Length',0.0,8.0)
    sepal_width=st.slider('Enter Sepal Width',0.0,8.0)
    petal_length=st.slider('Enter Petal Length',0.0,8.0)
    petal_width=st.slider('Enter Petal Width',0.0,8.0)

    result=""
    obj=""    
    if st.button("Predict"):
        file={"sepal_length":sepal_length,"sepal_width":sepal_width,"petal_length":petal_length,"petal_width":petal_width}
        url = 'http://127.0.0.1:8000/classify'
        x = requests.post(url, json = file)
        obj=x.json()
        result=obj[0]
    st.success(result)

if __name__== '__main__':
    main()