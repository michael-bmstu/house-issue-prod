import streamlit as st
from inference import *

st.title("Housse Issue")
train = read_train()
df = fill_na(train)

# get most popular
eng_types = list(map(lambda t: types_ru_en[t], train['Тип_жилья'].value_counts().index))
live_type = st.selectbox("Type of housing", eng_types)
city = st.selectbox("City", train['Город'].value_counts().index)
df.loc[0, ['Тип_жилья', 'Город']] = live_type, city
df.loc[0, ['Широта', 'Долгота', 'Индекс']] = get_geo(train, df['Город'].iloc[0])
st.write(f"Index: {df['Индекс'].values[0]:.0f}")
st.write(f"Latitude: {df['Широта'].values[0]:.5f}")
st.write(f"Longitude: {df['Долгота'].values[0]:.5f}")


bool_cols = ['Нлч_парковки', 'Нлч_почтового_ящика', 'Нлч_балкона', 'Нлч_террасы', 
             'Нлч_подвала', 'Нлч_гаража', 'Нлч_кондиционера']
eng_names = ['parking', 'mailbox', 'balcony', 'terrace', 'basement', 'garage', 'air conditioner']
with st.sidebar:
    st.write("Features")
    toggles = []
    for col, col_n  in zip(bool_cols, eng_names):
        toggles.append(st.toggle(col_n, value=True if df.loc[0, col] else False))

num_cols = ['Площадь', 'Размер_участка', 'Кво_комнат', 'Кво_спален']
eng_names = ['Square', 'Plot size', 'Rooms', 'Bedrooms']
num_inputs = []
for col, col_n in zip(num_cols, eng_names):
    num_inputs.append(st.number_input(col_n, 0, value=int(df.loc[0, col])))

for t, col in zip(toggles, bool_cols):
    df.loc[0, col] = 1 if t else 0

for inp, col in zip(num_inputs, num_cols):
    df.loc[0, col] = inp

df.loc[0, ['Тип_жилья', 'Город']] = live_type, city

df = fill_na(train, df)
df = add_features(df, train)

st.write(f"Recommended price for real estate:")
st.write(f"{pretty_num(str(predict(df)[0].round(2)))}$")

like = st.feedback("thumbs", )
if like is not None:
    st.write("You are awesome!!!")