import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Dublin_IRELAND.csv')

df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M')
df['Time'].dt.strftime('%Y-%m-%d %H:%M')
print(df)


df.to_csv('Dublin IRE AQI.csv', index=False)