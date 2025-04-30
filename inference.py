import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error as mape
from catboost import Pool
import streamlit as st

class MyMAPE: # MAPE for catboost log preds
    @staticmethod
    def mape(y_t, y_p):
        return mape(y_t, y_p)
    
    def get_final_error(self, error, weight):
        return error
    
    def is_max_optimal(self):
        # the larger metric value the better
        return False

    def evaluate(self, y_p, y_t, weight):
        score = self.mape(np.exp(y_t), np.exp(y_p[0]))
        return score, 0

types_en_ru = {
    "apartment": "квартира",
    "house": "дом",
    "duplex": "дуплекс",
    "villa": "вилла",
    "mansion": "особняк",
    "land": "земля",
    "miscellaneous": "разное",
    "land for development": "земля под застройку",
    "loft": "лофт",
    "farm": "ферма",
    "plot of land": "участок с землей",
    "life annuity": "пожизненная рента",
    "parking": "паркинг",
    "chalet": "шале",
    "flophouse": "ночлежка",
    "room": "комната",
    "house on the water": "дом на воде",
    "estate": "усадьба",
    "mill": "мельница",
    "workshop": "мастерская",
    "manor hotel": "отель-усадьба",
}
types_ru_en = {v: k for k, v in types_en_ru.items()}

def read_train(path="./data", drop_target=True):
    train = pd.read_csv(path + '/train.csv', index_col='id')
    if drop_target:
        target = 'Цена'
        train.drop(columns=target, inplace=True)
    return train

def get_geo(train, city):
    group = train.groupby(by='Город')
    lat, long = group[['Широта', 'Долгота']].mean().loc[city].values
    ind = group['Индекс'].apply(lambda x: x.mode()).loc[city].values[0]
    return lat, long, int(ind)

@st.cache_data
def get_pure(train):
    df = pd.DataFrame(columns=train.columns, index=[0])
    num_col = train.select_dtypes(include='number').columns
    df[num_col] = df[num_col].astype('float64')

    si_c = SimpleImputer(strategy='most_frequent')
    si_c.fit(train[['Тип_жилья', 'Город']])
    df[['Тип_жилья', 'Город']] = si_c.transform(df[['Тип_жилья', 'Город']])
    df.loc[0, ['Широта', 'Долгота', 'Индекс']] = get_geo(train, df['Город'].iloc[0])

    return df

@st.cache_data
def fill_na(train, df=None):
    df = get_pure(train) if df is None else df
    df.loc[0, ['Широта', 'Долгота', 'Индекс']] = get_geo(train, df['Город'].iloc[0])

    means = train.groupby(by='Тип_жилья')['Размер_участка'].mean().isna()
    zero_types = means.index[means.values]
    df.loc[df['Тип_жилья'].isin(zero_types), 'Размер_участка'] = 0
    
    means = train.groupby(by='Тип_жилья')['Расход_тепла'].mean().isna()
    zero_types = means.index[means.values]
    df.loc[df['Тип_жилья'].isin(zero_types), ['Расход_тепла', 'Кво_вредных_выбросов']] = 0

    mode = train.groupby(by='Тип_жилья')['Ктгр_энергоэффективности'].agg(lambda x: x.mode())
    empty_indices = [i for i, x in enumerate(mode.values) if isinstance(x, np.ndarray) and len(x) == 0]
    unk_types = mode.index[empty_indices]
    df.loc[df['Тип_жилья'].isin(unk_types), ['Ктгр_энергоэффективности', 'Ктгр_вредных_выбросов']] = 'unkown'
    
    mode = train.groupby(by='Тип_жилья')['Направление'].agg(lambda x: x.mode())
    empty_indices = [i for i, x in enumerate(mode.values) if isinstance(x, np.ndarray) and len(x) == 0]
    unk_types = mode.index[empty_indices]
    df.loc[df['Тип_жилья'].isin(unk_types), 'Направление'] = 'unkown'

    df = df.fillna({'Кво_фото': 0})
    df.loc[(df['Тип_жилья'] == 'комната') & df['Кво_спален'].isna(), 'Кво_спален'] = 1
    
    nan_num_cols = ['Площадь', 'Кво_комнат', 'Кво_спален', 'Размер_участка','Расход_тепла', 'Кво_вредных_выбросов']
    med_groups = train.groupby('Тип_жилья')[nan_num_cols].median()

    for col in nan_num_cols:
        df[col] = df.apply(
            lambda row: med_groups[col][row['Тип_жилья']] if pd.isna(row[col]) else row[col],
            axis=1)
    
    # simple filling
    si_n = SimpleImputer(strategy='median')
    si_c = SimpleImputer(strategy='most_frequent')
    
    num_cols = df.select_dtypes(include='number').columns
    si_n.fit(train[num_cols])
    df[num_cols] = si_n.transform(df[num_cols])
    
    cat_cols = df.select_dtypes(include='O').columns
    si_c.fit(train[cat_cols])
    df[cat_cols] = si_c.transform(df[cat_cols])
    
    return df

def clustring(df):
    with open("weights/kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)

    df['Кластер'] = kmeans.predict(df[['Широта', 'Долгота']])
    df['Кластер'] = df['Кластер'].astype("object")

    df_cd = kmeans.transform(df[['Широта', 'Долгота']])
    df_cd = pd.DataFrame(df_cd, columns=[f"Центр_{i}" for i in range(df_cd.shape[1])], index=df.index)
    df = df.join(df_cd)
    return df

def pca(df, train, col, top, model="pca_1"):
    with open(f"weights/{model}.pkl", "rb") as f:
        pca = pickle.load(f)
    df_pca = df[col]
    train_pca = train[col]
    df_pca_scaled = (df_pca - train_pca.mean(axis=0)) / train_pca.std(axis=0)
    df_pca = pca.transform(df_pca_scaled)

    df[top] = df_pca[:, [int(pca_t.split("_")[-1]) for pca_t in top]]
    return df

@st.cache_data
def add_features(df, train):
    def log_nan(row):
        row[row < 1] = 1
        return np.log(row)
    # df_ext = df.copy()
    cols_to_log = ['Площадь', 'Размер_участка', 'Расход_тепла', 'Кво_вредных_выбросов',]
    log_cols = ['Лог_' + col for col in cols_to_log]
    df[log_cols] = df[cols_to_log].apply(log_nan)
    train[log_cols] = train[cols_to_log].apply(log_nan)

    bool_cols = ['Нлч_парковки', 'Нлч_почтового_ящика', 'Нлч_балкона', 'Нлч_террасы', 'Нлч_подвала', 
                'Нлч_гаража', 'Нлч_кондиционера']
    df['Плюшки'] = df[bool_cols].sum(axis=1)

    df = clustring(df)
    train = clustring(train)
    df = pca(df, train, model="pca_1",
             col=['Площадь', 'Размер_участка', 'Расход_тепла', 'Кво_вредных_выбросов',
                'Лог_Площадь', 'Лог_Размер_участка', 'Лог_Расход_тепла', 'Лог_Кво_вредных_выбросов', 
                'Кво_комнат', 'Кво_спален', 'Кво_ванных',], 
             top=['PC_0', 'PC_1', 'PC_4', 'PC_8', 'PC_9'])
    df = pca(df, train, model="pca_2",
             col=['Центр_0', 'Центр_1','Центр_2', 'Центр_3', 'Центр_4', 'Центр_5', 'Центр_6', 
                  'Центр_7', 'Центр_8', 'Центр_9', 'Центр_10', 'Центр_11', 'Центр_12', 'Центр_13', 'Центр_14',],
             top=['PCц_0', 'PCц_5', 'PCц_1', 'PCц_4', 'PCц_9', 'PCц_3', 'PCц_2',
                    'PCц_11', 'PCц_10', 'PCц_14'])

    drop_cols = [
        'Последний_этаж', 'Верхний_этаж', 'Этаж',
        'Центр_2', 'Центр_3', 'Центр_4', 'Центр_6', 'Центр_7'
        ]
    return df.drop(columns=drop_cols)

def get_pred(models, x, strat='', weights=None):
    preds = np.zeros(x.shape[0])
    if strat == 'top' and weights is not None: 
        preds = np.exp(models[np.argmax(weights)].predict(x))
    else:
        for m, w in zip(models, weights):
            if strat == 'weighted' and weights is not None: preds += w * np.exp(m.predict(x))
            else: preds += np.exp(m.predict(x))

    if strat == '': preds /= len(models)
    return preds

@st.cache_data
def predict(df):
    models = np.load("./weights/bagging_models.npy", allow_pickle=True)
    weights = np.load("./weights/vote_weights.npy")
    cat_cols = list(df.select_dtypes(include='O').columns)
    test_pool = Pool(df, cat_features=cat_cols)

    return get_pred(models, test_pool, 'weighted', weights)

def pretty_num(s):
    s1, s2 = s.split(".")
    res = ""
    for i, sym in enumerate(s1[::-1], start=1):
        res += sym
        if i % 3 == 0:
            res += ' '
    return res[::-1].strip() + '.' + s2
    