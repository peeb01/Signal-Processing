import pandas as pd 

df = pd.read_csv('Dublin')

# df.fillna(0, inplace=True)

def calcul_aqi(pm25):
    breakpoints = [0, 12.0, 35.4, 55.4, 150.4, 250.4, 500.4]
    index_low = [0, 51, 101, 151, 201, 301, 501]
    index_high = [50, 100, 150, 200, 300, 500]

    for i in range(len(breakpoints) ):
        if breakpoints[i] <= pm25 <= breakpoints[i + 1]:
            aqi = ((index_high[i] - index_low[i]) / (breakpoints[i + 1] - breakpoints[i])) * (pm25 - breakpoints[i]) + index_low[i]
    return round(aqi)

print(df)

df['aqi'] = df['PM2.5'].apply(calcul_aqi)
# df['Date and Time'] = pd.to_datetime(df['Date and Time'], format='%m/%d/%Y %H:%M')
# df['Time'] = df['Date and Time'].dt.strftime('%Y-%m-%d %H:%M')
df = df[['date','PM2.5', 'aqi']]
# df.to_csv('grensasc_AQI.csv')
print(df)