import pandas as pd
#import matplotlib.plotly as py
#import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from collections import Counter

pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 150)
'''df = pd.read_excel("globalterrorismdb_0221dist.xlsx", index_col=0,
                      na_values=[''])

cols_of_interest = ["iyear",
                    'success',
                    'suicide',
                    'region_txt',
                     'attacktype1_txt',
                     'targtype1_txt',
                     'weaptype1_txt',
                     'propextent_txt',
                     'nkill',
                     'nperps',
                     'nwound',
                     'propvalue']

new_df = df[cols_of_interest]
print("Before Dropping NA Values: " + str(new_df.shape))

# Drop NA Values
newer_df = new_df.dropna()
print("After Dropping NA Values: " + str(newer_df.shape))
newer_df.head()

region_dummy = pd.get_dummies(newer_df["region_txt"])
attacktype1_dummy  = pd.get_dummies(newer_df["attacktype1_txt"])
targtype_txt_dummy  = pd.get_dummies(newer_df["targtype1_txt"])
prop_extent_txt_dummy  = pd.get_dummies(newer_df["propextent_txt"])
weaptype_txt_dummy  = pd.get_dummies(newer_df["weaptype1_txt"])

numerical_cols = ["iyear", 'success', 'suicide', 'nkill', 'nwound', 'propvalue', 'nperps']
final_df = pd.concat([newer_df, region_dummy, attacktype1_dummy, targtype_txt_dummy, prop_extent_txt_dummy, weaptype_txt_dummy], axis=1)
print(final_df.head())

# Remove Values Where 'nperps' equals -99 or Unknown
final_df = final_df[df["nperps"] != -99]

# Save Cleaned Data
final_df.to_csv("cleanDeadline_data.csv", encoding='utf-8')'''

train_data = pd.read_csv("cleanDeadline_data.csv", encoding = "ISO-8859-1")

train_data.drop('nkill', 1, inplace=True)
train_data.drop('nwound', 1, inplace=True)
train_data.drop('propvalue', 1, inplace=True)
train_data.drop('iyear', 1, inplace=True)

print(train_data.head())
