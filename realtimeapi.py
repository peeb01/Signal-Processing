import requests
import time
import pandas as pd
from datetime import datetime


def fetch_data_from_api():
    api_endpoint = 'https://api.airvisual.com/v2/city?city=Dublin&state=Leinster&country=Ireland&key=1b576f17-022b-4d4c-9ac9-61e6bdbbb665'
    try:
        response = requests.get(api_endpoint)   
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print("Error:", response.status_code)
            return None
    except Exception as e:
        print("An error occurred:", str(e))
        return None

# Set the end date and time (February 1, 2024, 00:00:00)
end_datetime = pd.to_datetime("2024-01-25 00:00:00")

column = ["City", "State", "Country", "Latitude", "Longitude", "AQI (US)", "Main (US)", "AQI (CN)", "Main (CN)", "Timestamp", "Temperature", "Pressure", "Humidity", "Wind Speed", "Wind Direction", "Weather Icon"]
data_df = pd.DataFrame(columns=column)
fetch_interval_seconds = 60

while pd.to_datetime("now") < end_datetime:
    data = fetch_data_from_api()
    if data:
        city = data['data']['city']
        state = data['data']['state']
        country = data['data']['country']
        coordinates = data['data']['location']['coordinates']
        pollution_data = data['data']['current']['pollution']
        weather_data = data['data']['current']['weather']

        data_df = data_df._append({
            "City": city,
            "State": state,
            "Country": country,
            "Latitude": coordinates[1],
            "Longitude": coordinates[0],
            "AQI (US)": pollution_data['aqius'],
            "Main (US)": pollution_data['mainus'],
            "AQI (CN)": pollution_data['aqicn'],
            "Main (CN)": pollution_data['maincn'],
            "Timestamp": pollution_data['ts'],
            "Temperature": weather_data['tp'],
            "Pressure": weather_data['pr'],
            "Humidity": weather_data['hu'],
            "Wind Speed": weather_data['ws'],
            "Wind Direction": weather_data['wd'],
            "Weather Icon": weather_data['ic']
        }, ignore_index=True)
        print("Now Data : " , datetime.now() , '\n')
        print(data_df)
        data_df.to_csv('AQI_PM2_5.csv', columns = column)

    # Wait for the specified interval before making the next request
    time.sleep(fetch_interval_seconds)

data_df.to_csv('data_until_2024-02-01.csv', index=False)
