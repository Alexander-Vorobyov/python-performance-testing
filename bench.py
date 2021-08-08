# This code was executed on different machines
# Этот код был запущен на разных компьютерах

from sklearn.datasets import make_regression
from xgboost.sklearn import XGBRegressor as XGBR
from time import time
import pandas as pd
from tqdm import tqdm

# Sizes of the dataset
# Размеры датасетов
sizes = [n*100 for n in range(1, 21)]

outp = pd.DataFrame(columns=['Size', 'Fit Time'])

for size in tqdm(sizes):
    start = time()
    X, y = make_regression(n_samples=size, n_features=size, n_informative=int(size*0.75))
    XGBR(nthread=-1).fit(X, y)
    end = time()
    outp.loc[len(outp)] = [size, round(end-start, 4)]
    
outp.to_excel('Bench results.xlsx', index=False)
print("Done")
