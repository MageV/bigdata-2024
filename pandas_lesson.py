import random
from datetime import datetime

import pandas as pd

def generate(rows_count=1):
    fmt = '%d-%m-%Y %H:%M:%S'
    countries_pool=['RU','KZ','US','CY','UA','AE']
    status_pool=[1,2,3]
    start = datetime.strptime('24-01-2023 10:00:00', fmt)
    end = datetime.strptime('24-07-2023 10:00:00', fmt)
    dates = [random.random() * (end - start) + start for _ in range(rows_count)]
    customers_count = [random.randint(0, 100) for _ in range(rows_count)]
    country=[countries_pool[random.randint(0, len(countries_pool)-1)] for _ in range(rows_count)]
    status = [status_pool[random.randint(0, len(status_pool)-1)] for _ in range(rows_count)]
    dataset = list(zip(dates, customers_count,country, status))
    print(dataset)
    df = pd.DataFrame(data=dataset, columns=['date', 'customers_count','country', 'status'])
    print(df)
    return df


generate(10)