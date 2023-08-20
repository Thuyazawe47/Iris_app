import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title='My Iris Test')
st.title("Iris Flower Classification")

@st.cache(allow_output_mutation=True)
def get_model():
    return joblib.load('iris_knn_model.joblib')

spl=st.text_input("Enter Sepal Length:","")
spw=st.text_input("Enter Sepal Width:","")
pel=st.text_input("Enter Petal Length:","")
pew=st.text_input("Enter Petal Width:","")

if st.button("Check Flower Type"):
    values=[spl,spw,pel,pew]
    num_values=[]
    for x in values:
        num_values.append(float(x))
    
    #2 dimension
    num_values=np.asarray(num_values).reshape(1,-1)
    predictions=get_model().predict(num_values)
    predictions=int(predictions)
    img='irir_flowers.png'
    if predictions==0:
        
        st.write("Flower is Iris-Sentosa") 
    elif predictions==1:
        st.write("Flower is Iris-verginia")
    else:
        st.write("Flower is Iris-versicolor")
    
    st.image(img,Caption="iris_flowers")
    
    
    
    
