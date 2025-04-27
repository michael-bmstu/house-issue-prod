import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from inference import *

st.title("Housse Issue")
train = read_train()
df = fill_na(train)

# get most popular
live_type = st.selectbox("Type of housing", train['Тип_жилья'].value_counts().index)
city = st.selectbox("City", train['Город'].value_counts().index)


bool_cols = ['Нлч_парковки', 'Нлч_почтового_ящика', 'Нлч_балкона', 'Нлч_террасы', 
             'Нлч_подвала', 'Нлч_гаража', 'Нлч_кондиционера']
with st.sidebar:
    toggles = []
    for col in bool_cols:
        col_n = col.split('_')[-1]
        if col_n[-1] == 'а':
            col_n = col_n[:-1]
        else:
            col_n = col_n[:-1] + 'a'
        toggles.append(st.toggle(col_n, value=True if df.loc[0, col] else False))

num_cols = ['Площадь', 'Размер_участка', 'Кво_комнат', 'Кво_спален', 'Кво_ванных']
num_inputs = []
for col in num_cols:
    num_inputs.append(st.number_input(' '.join(col.split('_')), 0, value=int(df.loc[0, col])))

for t, col in zip(toggles, bool_cols):
    df.loc[0, col] = 1 if t else 0

for inp, col in zip(num_inputs, num_cols):
    df.loc[0, col] = inp

df.loc[0, ['Тип_жилья', 'Город']] = live_type, city

df = fill_na(train, df)
df = add_features(df, train)

if st.button("Get recomend price"):
    st.write(f"{pretty_num(str(predict(df)[0].round(2)))}$")

like = st.feedback("thumbs")
if like is not None:
    st.write("You are awesome!!!")