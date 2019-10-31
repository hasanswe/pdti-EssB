import numpy as np
import mifs
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()

from sklearn.datasets import make_friedman1
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, confusion_matrix, classification_report, roc_curve, 
                             auc, roc_auc_score, matthews_corrcoef, make_scorer, recall_score, f1_score)

dataset = pd.read_excel(r"D:\Drug Target Interaction\Paper\Experiments\Ion Channel.xlsx")

dataset.head()

dataset.Target.value_counts()

dataset.Target.value_counts().plot.bar()

X = dataset.drop("Target", axis=1)
y = dataset["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#
for i in range(1,11):

    feat_selector = mifs.MutualInformationFeatureSelector('MRMR',k=i)
    feat_selector.fit(X_train, y_train)
    X_filtered = feat_selector.transform(X_train.values)
    feature_name = X_train.columns[feat_selector.ranking_]

    print(feature_name)
#

    X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, 5, step=1)
    selector = selector.fit(X, y)
    selector.support_ 
    array([ ....................])
     selector.ranking_
     array([................])

#
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
   feat_importances = pd.Series(model.feature_importances_, index=X.columns)
   feat_importances.nlargest(10).plot(kind='barh')
   plt.show()

#

clf_lr = LogisticRegression(C=100)
rfe = EnsRFS(estimator=clf_lr, n_features_to_select=1)
rfe.fit(X, y)

ranking = rfe.ranking_.reshape(35, 35)

plt.figure(figsize=(10, 10))
plt.matshow(ranking, cmap="Greens", fignum=1)
plt.colorbar()
plt.title("Ranking of pixels with EnsRFS", fontsize=18)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()
