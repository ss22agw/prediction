import pandas as pd
import numpy as np
import math
import seaborn as sns
from sklearn import preprocessing,metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

df = pd.read_csv('diabetes_prediction_dataset.csv')

df

df.info()

df.isna().sum()

print("Initial shape:", df.shape)
print("Duplicates in dataset:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("New shape:", df.shape)
print("Duplicates in dataset:", df.duplicated().sum())

plt.hist(df['age'],bins=10);

plt.xlabel('Age')
plt.ylabel('count')

plt.title('Histogram of Age')

sns.countplot(x=df['diabetes'])

sns.countplot(x=df['gender'],hue=df['diabetes'])

df.describe()

df.info()

continuous_features = list(set(['age','bmi','HbA1c_level','blood_glucose_level']))
continuous_features.sort()
continuous_features

for feature in continuous_features:
    fig,axs = plt.subplots(figsize=(22,9))
    sns.histplot(df[df['diabetes']==0][feature],color = 'red')
    sns.histplot(df[df['diabetes']==1][feature],color = 'blue')

    plt.legend([0,1],loc='upper right')
    plt.show()

for i in range(len(continuous_features)):
    feature = continuous_features[i]
    plt.figure(figsize = (10, 5))
    sns.boxplot(x='diabetes',y=continuous_features[i],data=df);

import scipy.stats as stats
for feature in continuous_features:
    print('---------------------------------')
    print('T-Test for::',feature)

    statt,p=stats.ttest_ind(df[feature][df['diabetes'] == 0],
                    df[feature][df['diabetes'] == 1])

    if p<.05:
        result="Mean value of for both target Condition is different"
    else:
        result="Mean value of for both target Condition is same"
    print('--> P-value is',p,"\n-->",result,"\n")

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot = True);

categorical_features = list(set(df.columns) -set(continuous_features)-set(['diabetes']))
categorical_features.sort()
categorical_features

for col in categorical_features:
    a= df[col].nunique()
    b= df[col].unique()
    print(col)
    print(a)
    print(b)

df["smoking_history"] = df["smoking_history"].replace("ever", "never")
df["smoking_history"] = df["smoking_history"].replace("not current", "former")

for feature in categorical_features:
    g = sns.catplot(x=feature, col='diabetes', kind='count', data=df, sharey=False)

from scipy.stats import chi2_contingency

for col in categorical_features:
    data_crosstab = pd.crosstab(df['diabetes'], df[col],)
    print(data_crosstab,"\n")
    c, p, dof, expected = chi2_contingency(data_crosstab)
    if p<.05:
        result="There is a significant association between these variables "
    else:
        result="There is no association those variables"
    print('--> P-value is',p,"\n-->",'chi2 value is',c,"\n-->",result,"\n")

plt.figure(figsize=(15,8))
df.boxplot(column=list(continuous_features))
plt.show()

df1=df.copy()

df1

categorical_features

df1 = pd.get_dummies(df,columns=categorical_features,drop_first=True)

df1

X = df1.drop(columns =['diabetes'])
y = df1['diabetes']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

X_train

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()

cols = list(continuous_features)
X_train[cols] = std_scaler.fit_transform(X_train[cols])

X_train

y_train

X_test[cols] = std_scaler.transform(X_test[cols])

X_test

y_test

def plotRocAuc(model, X, y):
    probabilities = model.predict_proba(X)
    probabilities = probabilities[:, 1]

    fpr, tpr, thresholds = roc_curve(y, probabilities)

    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.plot(fpr, tpr, marker='.')
    plt.text(0.75, 0.25, "AUC: " + str(round(roc_auc_score(y, probabilities),2)))

    plt.show()

clf_LR= LogisticRegression(random_state=11).fit(X_train, y_train)
pred = clf_LR.predict(X_test)

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, pred)))
print("Classification report:\n")
print(classification_report(y_test,pred))

print("Confusion Matrix:")
ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap='YlOrRd')
plt.show()

print("ROC curve:")
plotRocAuc(clf_LR,X_test,y_test)

from sklearn.naive_bayes import GaussianNB


clf_gnb = GaussianNB()
pred = clf_gnb.fit(X_train, y_train).predict(X_test)

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, pred)))
print("Classification report:\n")
print(classification_report(y_test,pred))

print("Confusion Matrix:")
ConfusionMatrixDisplay.from_predictions(y_test,pred, cmap='YlOrRd')
plt.show()

print("ROC curve:")
plotRocAuc(clf_gnb,X_test,y_test)


from sklearn.tree import DecisionTreeClassifier


clf_dtc = DecisionTreeClassifier(criterion='gini', max_depth=20, random_state=0)
clf_dtc.fit(X_train, y_train)

pred = clf_dtc.predict(X_test)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, pred)))
print("Classification report:\n")
print(classification_report(y_test,pred))

print("Confusion Matrix:")
ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap='YlOrRd')
plt.show()

print("ROC curve:")
plotRocAuc(clf_dtc,X_test,y_test)

from sklearn.ensemble import  RandomForestClassifier


clf_RF= RandomForestClassifier().fit(X_train, y_train)

pred = clf_RF.predict(X_test)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, pred)))
print("Classification report:\n")
print(classification_report(y_test,pred))

print("Confusion Matrix:")
ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap='YlOrRd')
plt.show()

print("ROC curve:")
plotRocAuc(clf_RF,X_test,y_test)

import xgboost as xgb


clf_xgb= xgb.XGBClassifier().fit(X_train, y_train)

pred = clf_xgb.predict(X_test)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, pred)))
print("Classification report:\n")
print(classification_report(y_test,pred))

print("Confusion Matrix:")
ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap='YlOrRd')
plt.show()

print("ROC curve:")
plotRocAuc(clf_xgb,X_test,y_test)

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

smt = SMOTE(k_neighbors=200, random_state=2)
X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)

smt_enn = SMOTEENN(random_state=2)
X_train_smt_enn, y_train_smt_enn = smt_enn.fit_resample(X_train, y_train)

smt_tom = SMOTETomek(random_state=2)
X_train_smt_tom, y_train_smt_tom = smt_tom.fit_resample(X_train, y_train)

import xgboost as xgb


clf_xgb= xgb.XGBClassifier().fit(X_train_smt_enn, y_train_smt_enn)

pred = clf_xgb.predict(X_test)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, pred)))
print("Classification report:\n")
print(classification_report(y_test,pred))

print("Confusion Matrix:")
ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap='YlOrRd')
plt.show()

print("ROC curve:")
plotRocAuc(clf_xgb,X_test,y_test)



result = pd.DataFrame()

result['Model'] = ['LR', 'NB', 'DT','RF','XGB','XGB_SMOTE']
result['Recall_macro_avg'] = [0.81,0.78, 0.85, 0.83, 0.84, 0.88]

result
