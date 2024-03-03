import pandas as pd 


df1 = pd.read_csv('Dataset\D9.csv')
df2 = pd.read_csv('Dataset\D8.csv')
df3 = pd.read_csv('Dataset\D7.csv')
df4 = pd.read_csv('Dataset\D6.csv')
df5 = pd.read_csv('Dataset\D5.csv')
df6 = pd.read_csv('Dataset\D4.csv')
df7 = pd.read_csv('Dataset\D3.csv')
df8 = pd.read_csv('Dataset\D2.csv')
df9 = pd.read_csv('Dataset\D1.csv')


df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9], axis=0)
df.to_csv('Dublin.csv')