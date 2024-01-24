import random
from datetime import datetime

import pandas as pd

fmt = '%d-%m-%Y %H:%M:%S'
start=datetime.strptime('24-01-2023 10:00:00',fmt)
end=datetime.strptime('24-07-2023 10:00:00',fmt)
dates=[random.random() * (end - start) + start for _ in range(10)]
customers_count=[random.randint(0,100) for _ in range(10)]
status=[random.randint(0,1) for _ in range(10)]
dataset=list(zip(dates,customers_count,status))
print(dataset)
df=pd.DataFrame(data=dataset,columns=['date','customers_count','status']);
print(df)