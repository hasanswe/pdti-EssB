import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, confusion_matrix, classification_report, roc_curve, 
                             auc, roc_auc_score, matthews_corrcoef, make_scorer, recall_score, f1_score)

dataset = pd.read_excel(r"D:\Drug Target Interaction\Datasets\Main datasets\Experiments\All Dataset\Experiments\Enzyme Drug Target Pair Datasets\Enzyme.xlsx")

dataset.head()

dataset.shape

dataset.Target.value_counts()

dataset.Target.value_counts().plot.bar()

X = dataset.drop("Target", axis=1)
y = dataset["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#SVM.SVCÂ

clf_svc = SVC(probability=True)
pred_svc = clf_svc.fit(X_train, y_train).predict(X_test)
acc = accuracy_score(pred_svc, y_test)
acc

svc_pred_proba1 = clf_svc.predict_proba(X_test)
svc_pred_proba1 = svc_pred_proba1[:, 1]
cm = confusion_matrix(pred_svc, y_test)
cm

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

true_positive, false_positive, false_negative, true_negative

print(classification_report(pred_svc, y_test))

sensitivity = (true_positive / (true_positive + false_negative))
specificity = (true_negative / (true_negative + false_positive))
sensitivity, specificity

auc_svc_1 = roc_auc_score(y_test, svc_pred_proba1)
print('AUC value of SVM: %.4f' % auc_svc_1)

f_measure = ((2 * true_positive)/ ((2 * true_positive) + false_positive + false_negative))
f_measure

mcc = matthews_corrcoef(pred_svc, y_test)
mcc

# Result Summary

{
    "AUC Score" : round(auc_svc_1, 4),
    "Accuracy" : round(acc, 4),
    "Sensitivity" : round(sensitivity, 4),
    "Specificity": round(specificity, 4),
    "MCC" : round(mcc, 4),
    "F1 Measure" : round(f_measure, 4)
}

# Logistic Regression
 

clf_lr = LogisticRegression(C=100)
pred_lr = clf_lr.fit(X_train, y_train).predict(X_test)
acc = accuracy_score(pred_lr, y_test)
acc

lr_pred_proba1 = clf_lr.predict_proba(X_test)
lr_pred_proba1 = lr_pred_proba1[:, 1]

cm = confusion_matrix(pred_lr, y_test)
cm

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

print(true_positive, false_positive, false_negative, true_negative)

print(classification_report(pred_lr, y_test))

sensitivity = (true_positive / (true_positive + false_negative))
specificity = (true_negative / (true_negative + false_positive))
sensitivity, specificity

auc_lr_1 = roc_auc_score(y_test, lr_pred_proba1)
print('AUC value of Logistic Regression: %.4f' % auc_lr_1)
f_measure = ((2 * true_positive)/ ((2 * true_positive) + false_positive + false_negative))
f_measure
mcc = matthews_corrcoef(pred_lr, y_test)
mcc

# Result Summary

{
    "AUC Score" : round(auc_lr_1, 4),
    "Accuracy" : round(acc, 4),
    "Sensitivity" : round(sensitivity, 4),
    "Specificity": round(specificity, 4),
    "MCC" : round(mcc, 4),
    "F1 Measure" : round(f_measure, 4)
}

# Random Forest 

clf_rf = RandomForestClassifier(random_state=10)
pred_rf = clf_rf.fit(X_train, y_train).predict(X_test)
acc = accuracy_score(pred_rf, y_test)
acc

rf_pred_proba1 = clf_rf.predict_proba(X_test)
rf_pred_proba1 = rf_pred_proba1[:, 1]

cm = confusion_matrix(pred_rf, y_test)
print(cm)

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

true_positive, false_positive, false_negative, true_negative

print(classification_report(pred_rf, y_test))

sensitivity = (true_positive / (true_positive + false_negative))
specificity = (true_negative / (true_negative + false_positive))
sensitivity, specificity

auc_rf_1 = roc_auc_score(y_test, rf_pred_proba1)
print('AUC value of Random Forest: %.4f' % auc_rf_1)

f_measure = ((2 * true_positive)/ ((2 * true_positive) + false_positive + false_negative))
f_measure
mcc = matthews_corrcoef(pred_rf, y_test)
mcc

# Result Summary

{
    "AUC Score" : round(auc_rf_1, 4),
    "Accuracy" : round(acc, 4),
    "Sensitivity" : round(sensitivity, 4),
    "Specificity": round(specificity, 4),
    "MCC" : round(mcc, 4),
    "F1 Measure" : round(f_measure, 4)
}

# XGBoost

clf_xgb = XGBClassifier(random_state=10, n_estimators=200, max_depth=10)
pred_xgb = clf_xgb.fit(X_train, y_train).predict(X_test)
acc = accuracy_score(pred_xgb, y_test)
acc

xgb_pred_proba1 = clf_xgb.predict_proba(X_test)
xgb_pred_proba1 = xgb_pred_proba1[:, 1]

cm = confusion_matrix(pred_xgb, y_test)
print(cm)

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

true_positive, false_positive, false_negative, true_negative

print(classification_report(pred_xgb, y_test))

sensitivity = (true_positive / (true_positive + false_negative))
specificity = (true_negative / (true_negative + false_positive))
sensitivity, specificity
auc_xgb_1 = roc_auc_score(y_test, xgb_pred_proba1)
print('AUC value of XGBoost: %.4f' % auc_xgb_1)
f_measure = ((2 * true_positive)/ ((2 * true_positive) + false_positive + false_negative))
f_measure

mcc = matthews_corrcoef(pred_xgb, y_test)
mcc

# Result Summary

{
    "AUC Score" : round(auc_xgb_1, 4),
    "Accuracy" : round(acc, 4),
    "Sensitivity" : round(sensitivity, 4),
    "Specificity": round(specificity, 4),
    "MCC" : round(mcc, 4),
    "F1 Measure" : round(f_measure, 4)
}

# ROC Curve

fpr_1, tpr_1, _ = roc_curve(y_test, svc_pred_proba1)
fpr_2, tpr_2, _ = roc_curve(y_test, lr_pred_proba1)
fpr_3, tpr_3, _ = roc_curve(y_test, rf_pred_proba1)
fpr_4, tpr_4, _ = roc_curve(y_test, xgb_pred_proba1)

roc_auc = dict()

roc_auc[0] = auc(fpr_1, tpr_1)
roc_auc[1] = auc(fpr_2, tpr_2)
roc_auc[2] = auc(fpr_3, tpr_3)
roc_auc[3] = auc(fpr_4, tpr_4)

roc_auc

sns.set_style("white")
plt.figure(figsize=(10, 6))
plt.figure(dpi=600)

plt.plot(fpr_1, tpr_1, color = "blue" , label = "Support Vector Machine - %0.2f" % roc_auc[0], lw=3)
plt.plot(fpr_2, tpr_2, color = "red" , label = "Logistic Regression - %0.2f" % roc_auc[1], lw=3)
plt.plot(fpr_3, tpr_3, color = "coral" , label = "Random Forest - %0.2f" % roc_auc[2], lw=3)
plt.plot(fpr_4, tpr_4, color = "#4caf50" , label = "XGBoost - %0.2f" % roc_auc[3], lw=3)

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Enzyme - [A, X, Y, Z]', fontsize=22)
plt.legend(loc="lower right", fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()


#Enzyme

dataset2 = pd.read_excel(r"D:\Drug Target Interaction\Datasets\Main datasets\Experiments\All Dataset\Experiments\Enzyme Drug Target Pair Datasets\Enzyme MSF+DPC.xlsx")

dataset2.head()
dataset2.shape
dataset2.Target.value_counts()

dataset2.Target.value_counts().plot.bar()
X = dataset2.drop("Target", axis=1)
y = dataset2["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


#SVM Classifier



clf_svc = SVC(probability=True)
pred_svc = clf_svc.fit(X_train, y_train).predict(X_test)
acc = accuracy_score(pred_svc, y_test)
acc

svc_pred_proba2 = clf_svc.predict_proba(X_test)
svc_pred_proba2 = svc_pred_proba2[:, 1]


cm = confusion_matrix(pred_svc, y_test)
cm

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

true_positive, false_positive, false_negative, true_negative

print(classification_report(pred_svc, y_test))

sensitivity = (true_positive / (true_positive + false_negative))
specificity = (true_negative / (true_negative + false_positive))
sensitivity, specificity

auc_svc_2 = roc_auc_score(y_test, svc_pred_proba2)
print('AUC value of SVM: %.4f' % auc_svc_2)

f_measure = ((2 * true_positive)/ ((2 * true_positive) + false_positive + false_negative))
f_measure

mcc = matthews_corrcoef(pred_svc, y_test)
mcc


#Result Summary

{
    "AUC Score" : round(auc_svc_2, 4),
    "Accuracy" : round(acc, 4),
    "Sensitivity" : round(sensitivity, 4),
    "Specificity": round(specificity, 4),
    "MCC" : round(mcc, 4),
    "F1 Measure" : round(f_measure, 4)
}


#Logistic Regression

clf_lr = LogisticRegression(C=20)
pred_lr = clf_lr.fit(X_train, y_train).predict(X_test)
acc = accuracy_score(pred_lr, y_test)
acc 

lr_pred_proba2 = clf_lr.predict_proba(X_test)
lr_pred_proba2 = lr_pred_proba2[:, 1]

cm = confusion_matrix(pred_lr, y_test)
cm


true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

true_positive, false_positive, false_negative, true_negative

print(classification_report(pred_lr, y_test))

sensitivity = (true_positive / (true_positive + false_negative))
specificity = (true_negative / (true_negative + false_positive))
sensitivity, specificity

auc_lr_2 = roc_auc_score(y_test, lr_pred_proba2)
print('AUC value of Logistic Regression: %.4f' % auc_lr_2)

f_measure = ((2 * true_positive)/ ((2 * true_positive) + false_positive + false_negative))
f_measure

mcc = matthews_corrcoef(pred_lr, y_test)
mcc

{
    "AUC Score" : round(auc_lr_2, 4),
    "Accuracy" : round(acc, 4),
    "Sensitivity" : round(sensitivity, 4),
    "Specificity": round(specificity, 4),
    "MCC" : round(mcc, 4),
    "F1 Measure" : round(f_measure, 4)
}


#Random Forest

clf_rf = RandomForestClassifier(random_state=10)
pred_rf = clf_rf.fit(X_train, y_train).predict(X_test)
acc = accuracy_score(pred_rf, y_test)
acc

rf_pred_proba2 = clf_rf.predict_proba(X_test)
rf_pred_proba2 = rf_pred_proba2[:, 1]

cm = confusion_matrix(pred_rf, y_test)
print(cm)

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

true_positive, false_positive, false_negative, true_negative


print(classification_report(pred_rf, y_test))

sensitivity = (true_positive / (true_positive + false_negative))
specificity = (true_negative / (true_negative + false_positive))
sensitivity, specificity
auc_rf_2 = roc_auc_score(y_test, rf_pred_proba2)
print('AUC value of Random Forest: %.4f' % auc_rf_2)

f_measure = ((2 * true_positive)/ ((2 * true_positive) + false_positive + false_negative))
f_measure
mcc = matthews_corrcoef(pred_rf, y_test)
mcc


#Result Summary

{
    "AUC Score" : round(auc_rf_2, 4),
    "Accuracy" : round(acc, 4),
    "Sensitivity" : round(sensitivity, 4),
    "Specificity": round(specificity, 4),
    "MCC" : round(mcc, 4),
    "F1 Measure" : round(f_measure, 4)
}


#XGBoost

clf_xgb = XGBClassifier(random_state=10, n_estimators=200, max_depth=10)
pred_xgb = clf_xgb.fit(X_train, y_train).predict(X_test)
acc = accuracy_score(pred_xgb, y_test)
acc
xgb_pred_proba2 = clf_xgb.predict_proba(X_test)
xgb_pred_proba2 = xgb_pred_proba2[:, 1]

cm = confusion_matrix(pred_xgb, y_test)
print(cm)

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

true_positive, false_positive, false_negative, true_negative
print(classification_report(pred_xgb, y_test))

sensitivity = (true_positive / (true_positive + false_negative))
specificity = (true_negative / (true_negative + false_positive))
sensitivity, specificity
auc_xgb_2 = roc_auc_score(y_test, xgb_pred_proba2)
print('AUC value of XGBoost: %.4f' % auc_xgb_2)
f_measure = ((2 * true_positive)/ ((2 * true_positive) + false_positive + false_negative))
f_measure

mcc = matthews_corrcoef(pred_xgb, y_test)
mcc

{
    "AUC Score" : round(auc_xgb_2, 4),
    "Accuracy" : round(acc, 4),
    "Sensitivity" : round(sensitivity, 4),
    "Specificity": round(specificity, 4),
    "MCC" : round(mcc, 4),
    "F1 Measure" : round(f_measure, 4)
}


#ROC Curve

fpr_1, tpr_1, _ = roc_curve(y_test, svc_pred_proba2)
fpr_2, tpr_2, _ = roc_curve(y_test, lr_pred_proba2)
fpr_3, tpr_3, _ = roc_curve(y_test, rf_pred_proba2)
fpr_4, tpr_4, _ = roc_curve(y_test, xgb_pred_proba2)

roc_auc = dict()

roc_auc[0] = auc(fpr_1, tpr_1)
roc_auc[1] = auc(fpr_2, tpr_2)
roc_auc[2] = auc(fpr_3, tpr_3)
roc_auc[3] = auc(fpr_4, tpr_4)

roc_auc

sns.set_style("white")
plt.figure(figsize=(10, 6))
plt.figure(dpi=600)

plt.plot(fpr_1, tpr_1, color = "blue" , label = "Support Vector Machine - %0.2f" % roc_auc[0], lw=3)
plt.plot(fpr_2, tpr_2, color = "red" , label = "Logistic Regression - %0.2f" % roc_auc[1], lw=3)
plt.plot(fpr_3, tpr_3, color = "coral" , label = "Random Forest - %0.2f" % roc_auc[2], lw=3)
plt.plot(fpr_4, tpr_4, color = "#4caf50" , label = "XGBoost - %0.2f" % roc_auc[3], lw=3)

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Enzyme - [A, X, Y, Z]', fontsize=22)
plt.legend(loc="lower right", fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()