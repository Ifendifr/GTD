import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Configure notebook output
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"


# Function to impute either the median or mean
def fill_value(attr):
    fill = 0.0
    threshold = 3
    attr_clean = attr.dropna()
    attr_std = attr_clean.std()
    outliers = attr_clean[attr_clean > (threshold * attr_std)]

    if (outliers.count() > 0):
        fill = attr_clean.median()
    else:
        fill = attr_clean.mean()

    return fill


# Display up to 150 rows and columns
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 150)

# Set the figure size for plots
mpl.rcParams['figure.figsize'] = (14.6, 9.0)

# Set the Seaborn default style for plots
sns.set()

# Set the color palette
sns.set_palette(sns.color_palette("muted"))

# Load the preprocessed GTD dataset
gtd_df = pd.read_csv('gtd_preprocessed_95t016.csv', index_col = 0,
                      na_values=[''])
#print(gtd_df.info(verbose = True))

gtd_df.loc[gtd_df['weaptype1_txt'] ==
           'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)',
           'weaptype1_txt'] = 'Vehicle (non-explosives)'

gtd_df.loc[gtd_df['attacktype1_txt'] ==
           'Hostage Taking (Barricade Incident)',
           'attacktype1_txt'] = 'Hostage Taking (Barricade)'

# List of attributes that are categorical
cat_attrs = ['extended_txt', 'country_txt', 'region_txt', 'specificity', 'vicinity_txt',
             'crit1_txt', 'crit2_txt', 'crit3_txt', 'doubtterr_txt', 'multiple_txt',
             'success_txt', 'suicide_txt', 'attacktype1_txt', 'targtype1_txt',
             'targsubtype1_txt', 'natlty1_txt', 'guncertain1_txt', 'individual_txt',
             'claimed_txt', 'weaptype1_txt', 'weapsubtype1_txt', 'property_txt',
             'ishostkid_txt', 'INT_LOG_txt', 'INT_IDEO_txt', 'INT_MISC_txt', 'INT_ANY_txt']

for cat in cat_attrs:
    gtd_df[cat] = gtd_df[cat].astype('category')



# Impute each of the numeric attributes that contain missing values
gtd_df['nperpcap'] = gtd_df['nperpcap'].fillna(fill_value(gtd_df['nperpcap']))
gtd_df['nkill'] = gtd_df['nkill'].fillna(fill_value(gtd_df['nkill']))
gtd_df['nkillus'] = gtd_df['nkillus'].fillna(fill_value(gtd_df['nkillus']))
gtd_df['nkillter'] = gtd_df['nkillter'].fillna(fill_value(gtd_df['nkillter']))
gtd_df['nwound'] = gtd_df['nwound'].fillna(fill_value(gtd_df['nwound']))
gtd_df['nwoundus'] = gtd_df['nwoundus'].fillna(fill_value(gtd_df['nwoundus']))
gtd_df['nwoundte'] = gtd_df['nwoundte'].fillna(fill_value(gtd_df['nwoundte']))

print(gtd_df.info(verbose=True))

# Select the observations that contain null
ll_df = gtd_df[np.isnan(gtd_df.latitude)]
print(ll_df.shape)

# Chech how many observations have city set to Unknown
city_df = ll_df[(ll_df['city'] == "UNKNOWN")]
print(city_df['city'].value_counts())

# Remove observations containing missing missing values for latitude and longitude
gtd_clean = gtd_df.dropna().copy()
#gtd_clean.info(verbose = True)

# 297 iday attributes contain 0 to represent unknown, setting 1
gtd_clean.loc[gtd_clean['iday'] == 0, 'iday'] = 1
gtd_clean.loc[gtd_clean['imonth'] == 0, 'imonth'] = 1
gtd_clean['incident_date'] = (gtd_clean['iyear'].astype(str) + '-' +
                              gtd_clean['imonth'].astype(str) + '-' +
                              gtd_clean['iday'].astype(str))

gtd_clean['incident_date'] = pd.to_datetime(gtd_clean['incident_date'],
                                            format="%Y-%m-%d")
print(gtd_clean.info(verbose = True))
gtd_clean.to_csv("gtd_eda_95t016.csv", sep = ",")