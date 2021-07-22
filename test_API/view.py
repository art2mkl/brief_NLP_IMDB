
#shell : streamlit run view.py 

import streamlit as st
import requests
import pandas as pd
st.set_page_config(layout="wide")

st.title('Generateur IA de proposition de films')


st.subheader('Voir un dataset de film')
df = pd.DataFrame(requests.get('http://127.0.0.1:8000/line/').json())
df


st.subheader("Comment t'appelles-tu ?")
add_line=""
add_line = st.text_input("écris ton nom", "")
var = None
if add_line:
    var = add_line

submit = st.button('submit')

if submit:

    # params = (
    # ('data', var),
    # )
    #st.write(requests.post('http://127.0.0.1:8000/add/', params=params).text)

    if var is not None :
    #st.write(requests.post("http://127.0.0.1:8000/add/", params = {'data':var}).text)
        st.write(requests.post('http://127.0.0.1:8000/add/', params = {'data2':var}).text)
   

cols = st.beta_columns(3)
vars = [0,0,0]

cols[0].write(df[['films_1']])
choix_0 = cols[0].selectbox(f'topics {1}',[1,2,3])
if choix_0:
    vars[0] = choix_0

cols[1].write(df[['films_2']])
choix_1 = cols[1].selectbox(f'topics {2}',[1,2,3])
if choix_1:
    vars[1] = choix_1

cols[2].write(df[['films_3']])
choix_2 = cols[2].selectbox(f'topics {3}',[1,2,3])
if choix_2:
    vars[2] = choix_2


validation = st.button('validation')


if validation:
    if len(set(vars)) != 3:
        st.write('Mettez un choix unique')
    else :

        st.write(requests.post('http://127.0.0.1:8000/choice/', params = {
            'data_1':vars[0],
            'data_2':vars[1],
            'data_3':vars[2]
            }).text)
    
