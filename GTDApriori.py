import time
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 150)

# Set the figure size for plots
mpl.rcParams['figure.figsize'] = (14.6, 9.0)

# Set the Seaborn default style for plots
sns.set()

# Set the color palette
sns.set_palette(sns.color_palette("muted"))

# Load the preprocessed GTD dataset
gtd_df = pd.read_csv('gtd_eda_95t016.csv', index_col = 0, na_values=[''])
# Display a summary of the data frame
#gtd_df.info(verbose = True)

gtd_df.loc[gtd_df['attacktype1_txt'] == 'Unknown', 'attacktype1_txt'] = 'att_Unknown'
gtd_df.loc[gtd_df['gname'] == 'Unknown', 'gname'] = 'gna_Unknown'
gtd_df.loc[gtd_df['weaptype1_txt'] == 'Unknown', 'weaptype1_txt'] = 'weap_Unknown'

# List of attributes that are categorical
cat_attrs = ['extended_txt', 'country_txt', 'region_txt', 'specificity', 'vicinity_txt',
             'crit1_txt', 'crit2_txt', 'crit3_txt', 'doubtterr_txt', 'multiple_txt',
             'success_txt', 'suicide_txt', 'attacktype1_txt', 'targtype1_txt',
             'targsubtype1_txt', 'natlty1_txt', 'guncertain1_txt', 'individual_txt',
             'claimed_txt', 'weaptype1_txt', 'weapsubtype1_txt', 'property_txt',
             'ishostkid_txt', 'INT_LOG_txt', 'INT_IDEO_txt', 'INT_MISC_txt', 'INT_ANY_txt']

for cat in cat_attrs:
    gtd_df[cat] = gtd_df[cat].astype('category')

# Data time feature added during EDA
gtd_df['incident_date'] = pd.to_datetime(gtd_df['incident_date'])

# To prevent a mixed data type
gtd_df['gname'] = gtd_df['gname'].astype('str')

gtd_df.info(verbose = True)


gtd = gtd_df[['country_txt','region_txt','city','attacktype1_txt','targtype1_txt','targsubtype1_txt',
             'natlty1_txt','gname','weaptype1_txt']]




gtd.to_csv('GTDtest.txt', index=False, sep='\t', header=None, encoding='utf_8_sig')