import pandas as pd

df = pd.read_csv('datasets/percentile_rankings.csv')

df_cleaned = df.dropna(thresh=10)

df_cleaned.to_csv('datasets/cleaned_percentile_rankings.csv', index=False)
