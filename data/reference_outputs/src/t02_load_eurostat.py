import pandas as pd

selected = pd.read_csv('outputs/t01_selected_indicators.csv')
raw = pd.read_csv('inputs/eurostat/eurostat_city_values_sample.csv')
raw = raw[raw['indicator_id'].isin(selected['indicator_id'])]
columns = ['city_id', 'city_name', 'year', 'indicator_id', 'indicator_name', 'value', 'unit']
raw[columns].to_csv('outputs/t02_eurostat_raw.csv', index=False)
