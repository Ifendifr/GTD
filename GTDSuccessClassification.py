import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import os
from scipy import stats
from scipy.stats import norm
plt.style.use('ggplot')
sys.path.append(os.path.abspath("../../shared"))
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

import feature_selection, helpers, eda, preprocessing, metric_eval
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')


def preprocess(data, debug=False, no_label_encode=False):
    missing = preprocessing.percent_missing(data)
    data = data.drop(missing[missing['Percent'] >= .50].index, 1)

    txt_cols = [c for c in data.columns if c.lower()[-3:] == 'txt']
    data = data.drop(labels=txt_cols, axis=1)

    free_text_cols = ['motive', 'addnotes', 'propcomment', 'summary']
    citing_cols = ['scite1', 'scite2', 'scite3', 'dbsource']
    not_sure_cols = ['related', 'location', 'weapdetail', 'corp1']
    label_encode = ['provstate', 'related', 'location', 'weapdetail', 'corp1', 'city', 'gname',
                    'target1']  # <- Will need to learn feature extraction later for this

    # Label encoding features
    if not no_label_encode:
        from sklearn.preprocessing import LabelEncoder
        for f in label_encode:
            if f in data.columns:
                le = LabelEncoder()
                data[f] = data[f].fillna('None')
                data[f] = le.fit_transform(data[f])

    citing_cols.extend(free_text_cols)
    for col in citing_cols:
        if col in data.columns:
            data = data.drop(labels=col, axis=1)

    missing = preprocessing.percent_missing(data)
    missing_cat_cols = ['ransom', 'claimed', 'weapsubtyp1',
                        'targsubtype1', 'natlty1', 'guncertain1',
                        'ishostkid', 'specificy', 'latitude', 'longitude']
    for f in missing.index:
        # If not a categorical variable, its numeric, so take median
        if f not in missing_cat_cols:
            data[f] = data[f].fillna(data[f].median())

    for f in missing_cat_cols:
        if f in data.columns:
            data[f] = data[f].fillna(data[f].max())

    if debug:
        print(data.dtypes)
    return data



data = pd.read_csv('gtd.csv', encoding='ISO-8859-1')

# Number of data points in the minority class
dfs=data[data['success']==1]
dff=data[data['success']==0]
undersampled_data=pd.concat([dff, dfs.sample(len(dff))])

undersample_X = preprocess(undersampled_data)

undersample_y = undersample_X['success']
undersample_X = undersample_X.drop(labels='success',axis=1)
undersample_X.shape

dummy = DummyClassifier(strategy='most_frequent')
#cross_val_score(dummy, undersample_X, undersample_y, cv = 10)

reg_model = RandomForestClassifier(criterion='entropy', min_samples_split=50)
#np.array(cross_val_score(reg_model, undersample_X, undersample_y, cv = 10, n_jobs=-1)).mean()

data_X = preprocess(data)

data_y = data_X['success']
data_X = data_X.drop('success',1)

# Whole dataset
data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_X,data_y,
                                                                        test_size = 0.2, shuffle=True)


# Undersampled dataset
undersample_X_train, undersample_X_test, undersample_y_train, undersample_y_test = \
                                                    train_test_split(undersample_X,
                                                                     undersample_y,
                                                                     test_size = 0.2, shuffle=True)
undersample_rfc = RandomForestClassifier()
undersample_rfc.fit(undersample_X_train, undersample_y_train)
undersample_predictions = undersample_rfc.predict(undersample_X_test)
metric_eval.classification_report(undersample_y_test, undersample_predictions
                      , classes=['failed attack','success']
                      , title='Full dataset confusion matrix');
