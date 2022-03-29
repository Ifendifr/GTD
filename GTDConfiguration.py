import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"

'''# Number of rows and columns
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 150)
# Load 1995-2012 GTD
gtd_df1 = pd.read_csv('../data/gtd_95to12_0617dist.csv', low_memory=False, index_col = 0,
                      na_values=[''])

# Load 2013-2016 GTD
gtd_df2 = pd.read_csv('../data/gtd_13to16_0617dist.csv', low_memory=False, index_col = 0,
                      na_values=[''])

# Append the 2nd data frame to the first
gtd_df = gtd_df1.append(gtd_df2)

# Display a summary of the data frame
gtd_df.info(verbose = True)'''
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 150)
gtd_df = pd.read_excel("globalterrorismdb_0221dist.xlsx", index_col=0,
                      na_values=[''])
#data = gtd_df.head()#默认读取前5行的数据
#print("获取到所有的值:\n{0}".format(data))
#print(gtd_df.info(verbose = True))

count = gtd_df.isnull().sum()
percent = round(count / 201183 * 100, 2)
series = [count, percent]
result = pd.concat(series, axis=1, keys=['Count','Percent'])
#print(result.sort_values(by='Count', ascending=False))

target_attrs = result[result['Percent'] < 36.0]
keep_attrs = target_attrs.index.values

# The nperps attribute contain 18.91% blank values.  However, an additional 64.31% are
# coded (-99, -9) as unknown.
keep_attrs = keep_attrs[keep_attrs != 'nperps']


# Remove attributes that duplicate another attribute
keep_attrs = keep_attrs[keep_attrs != 'country']
keep_attrs = keep_attrs[keep_attrs != 'region']
keep_attrs = keep_attrs[keep_attrs != 'attacktype1']
keep_attrs = keep_attrs[keep_attrs != 'targtype1']
keep_attrs = keep_attrs[keep_attrs != 'targsubtype1']
keep_attrs = keep_attrs[keep_attrs != 'natlty1']
keep_attrs = keep_attrs[keep_attrs != 'weaptype1']
keep_attrs = keep_attrs[keep_attrs != 'weapsubtype1']

subset_df = gtd_df.loc[:, keep_attrs]
#print(subset_df.info(verbose = True))

# Categorical Variables
# ---------------------
subset_df['specificity'].fillna(-1, inplace=True)

subset_df.loc[subset_df['vicinity'] == -9, 'vicinity'] = -1

subset_df.loc[subset_df['doubtterr'] == -9, 'doubtterr'] = -1

subset_df['targsubtype1_txt'].fillna('UNKNOWN', inplace=True)

subset_df['natlty1_txt'].fillna('UNKNOWN', inplace=True)

subset_df['guncertain1'].fillna(-1, inplace=True)

subset_df['claimed'].fillna(-1, inplace=True)
subset_df.loc[subset_df['claimed'] == -9, 'claimed'] = -1

subset_df['weapsubtype1_txt'].fillna('UNKNOWN', inplace=True)

subset_df.loc[subset_df['property'] == -9, 'property'] = -1

subset_df['ishostkid'].fillna(-1, inplace=True)
subset_df.loc[subset_df['ishostkid'] == -9, 'ishostkid'] = -1

subset_df.loc[subset_df['INT_LOG'] == -9, 'INT_LOG'] = -1

subset_df.loc[subset_df['INT_IDEO'] == -9, 'INT_IDEO'] = -1

subset_df.loc[subset_df['INT_MISC'] == -9, 'INT_MISC'] = -1

subset_df.loc[subset_df['INT_ANY'] == -9, 'INT_ANY'] = -1


# Numeric Variables
# -----------------
subset_df.loc[subset_df['nperpcap'] == -9, 'nperpcap'] = np.nan
subset_df.loc[subset_df['nperpcap'] == -99, 'nperpcap'] = np.nan


# Text Variables
# --------------
subset_df['provstate'].fillna('UNKNOWN', inplace=True)
subset_df['city'].fillna('UNKNOWN', inplace=True)
subset_df.loc[subset_df['city'] == 'Unknown', 'city'] = 'UNKNOWN'
subset_df['summary'].fillna('UNKNOWN', inplace=True)
subset_df['corp1'].fillna('UNKNOWN', inplace=True)
subset_df['target1'].fillna('UNKNOWN', inplace=True)
subset_df['scite1'].fillna('UNKNOWN', inplace=True)

# Map the codes to labels
ynu_map = {1: 'YES', 0: 'NO', -1: 'UKNOWN'}

# List of target attributes to map
ynu_attrs = ['extended', 'vicinity', 'crit1', 'crit2', 'crit3', 'doubtterr', 'multiple',
             'success', 'suicide', 'guncertain1', 'individual', 'claimed', 'property',
             'ishostkid', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY']

# Iterate over each target attribute and map it
for att in ynu_attrs:
    att_txt = att + '_txt'
    subset_df[att_txt] = subset_df[att].map(ynu_map)

# Get the list of attributes, dropping the coded for labeled attributes
final_attrs = []

for attr in subset_df.columns.values:
    if attr not in ynu_attrs:
        final_attrs.append(attr)

subset_df2 = subset_df.loc[:, final_attrs]
print(subset_df2.info(verbose=True))

subset_df2.to_csv("gtd_preprocessed_95t016.csv", sep = ",")

