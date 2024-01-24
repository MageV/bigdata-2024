import random
from datetime import datetime

import pandas as pd
import numpy as np


def generate(rows_count=1):
    fmt = '%d-%m-%Y %H:%M:%S'
    countries_pool = ['RU', 'KZ', 'US', 'CY', 'UA', 'AE']
    status_pool = [1, 2, 3]
    dates_rng = pd.date_range(start='1/24/2023', end='7/24/2023')
    dates = [dates_rng[random.randint(0, dates_rng.values.size-1)] for _ in range(rows_count)]
    customers_count = np.random.randint(low=25, high=1000, size=dates_rng.values.size)
    country = [countries_pool[random.randint(0, len(countries_pool) - 1)] for _ in range(rows_count)]
    status = [status_pool[random.randint(0, len(status_pool) - 1)] for _ in range(rows_count)]
    dataset = list(zip(dates, customers_count, country, status))
    print(dataset)
    df = pd.DataFrame(data=dataset, columns=['date', 'customers_count', 'country', 'status'])
    print(df)
    return df


frame=generate(10)
frame.to_excel('data/customers.xlsx',index=False,header=True,engine='openpyxl')
frame2=pd.read_excel('data/customers.xlsx',index_col='date')
print(frame2)
