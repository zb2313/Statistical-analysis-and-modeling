import pandas as pd
from collections import Counter
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats


def check_repeat(f1):
    print(f1.shape)
    f1.drop_duplicates(subset='SMILES')  # 判断是否有行重复
    print(f1.shape)
    columns = f1.columns.values  # 判断是否有列重复
    print(len(columns))
    np.unique(columns)
    print(len(columns))


# 缺失值判断
def check_null(f1):
    print(f1.columns[f1.isnull().any()].tolist())


def get_data():
    f1 = pd.read_csv('f1.csv')
    fz = pd.DataFrame()
    for attr in f1.columns[1:]:
        fz[attr + '_zscore'] = stats.zscore(f1[attr])
    list = []
    for index, row in fz.iterrows():
        for attr in fz.columns:
            if abs(row[attr]) >= 3:
                list.append(index)
    result = Counter(list)
    list2 = result.most_common(52)
    for i in list2:
        f1.drop(index=[i[0]], inplace=True)
        fz.drop(index=[i[0]], inplace=True)
    return f1

