import pandas as pd
data = [['impact', 'JANUARY 1, 2021'], ['safaricom', 'FEBRUARY 21, 2021']]

df = pd.DataFrame(data, columns=['News', 'Date'])
df.to_csv('test.csv', index=False)
