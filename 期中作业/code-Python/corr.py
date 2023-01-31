import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn
import statsmodels.api as sa
# spearman相关分析
from scipy import stats
from collections import Counter


def get_attributes(f1):
    # print(f1.corr())
    # print(f1.corr('spearman'))
    # seaborn.heatmap(f1.corr('spearman'))
    # plt.show()
    f1_corr = f1.corr('spearman')['pIC50'].dropna().sort_values()  # 相关系数排序
    print(f1_corr)
    f1_corr_sort = f1_corr.abs().sort_values()
    # print(f1_corr_sort[-21:])
    print(f1_corr_sort[-21:-1])
    return f1_corr_sort[-21:].index

# 灰色关联分析？
