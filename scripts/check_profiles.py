import pandas as pd
df = pd.read_json('data/user_profiles.json')
print(df.columns.tolist())
print(df.head(3).to_string())