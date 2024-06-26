import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as mso
import seaborn as sns
import warnings
import os
import scipy

from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



df = pd.read_csv(r"C:\Users\sasuk\Downloads\archive\train_u6lujuX_CVtuZ9i.csv")
df.head()

# the 13 columns are readable. It also can be seen that there are 614 observations in the data set.
print(df.shape)
# (614, 13)


# print(df.Loan_ID.value_counts(dropna=False))

# print(df.Gender.value_counts(dropna=False))

# sns.countplot(x="Gender", data=df, palette="hls")
# plt.show()

# countMale = len(df[df.Gender == 'Male'])
# countFemale = len(df[df.Gender == 'Female'])
# countNull = len(df[df.Gender.isnull()])
# print("Percentage of Male applicant: {:.2f}%".format((countMale / (len(df.Gender))*100)))
# print("Percentage of Female applicant: {:.2f}%".format((countFemale / (len(df.Gender))*100)))
# print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Gender))*100)))

# print(df.Married.value_counts(dropna=False))

# sns.countplot(x="Married", data=df, palette="Paired")
# plt.show()

# countMarried = len(df[df.Married == 'Yes'])
# countNotMarried = len(df[df.Married == 'No'])
# countNull = len(df[df.Married.isnull()])
# print("Percentage of married: {:.2f}%".format((countMarried / (len(df.Married))*100)))
# print("Percentage of Not married applicant: {:.2f}%".format((countNotMarried / (len(df.Married))*100)))
# print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Married))*100)))

# print(df.Education.value_counts(dropna=False))

# sns.countplot(x="Education", data=df, palette="rocket")
# plt.show()

# countGraduate = len(df[df.Education == 'Graduate'])
# countNotGraduate = len(df[df.Education == 'Not Graduate'])
# countNull = len(df[df.Education.isnull()])
# print("Percentage of graduate applicant: {:.2f}%".format((countGraduate / (len(df.Education))*100)))
# print("Percentage of Not graduate applicant: {:.2f}%".format((countNotGraduate / (len(df.Education))*100)))
# print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Education))*100)))

# print(df.Self_Employed.value_counts(dropna=False))

# sns.countplot(x="Self_Employed", data=df, palette="crest")
# plt.show()

# countNo = len(df[df.Self_Employed == 'No'])
# countYes = len(df[df.Self_Employed == 'Yes'])
# countNull = len(df[df.Self_Employed.isnull()])
# print("Percentage of Not self employed: {:.2f}%".format((countNo / (len(df.Self_Employed))*100)))
# print("Percentage of self employed: {:.2f}%".format((countYes / (len(df.Self_Employed))*100)))
# print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Self_Employed))*100)))

# print(df.Credit_History.value_counts(dropna=False))

# sns.countplot(x="Credit_History", data=df, palette="viridis")
# plt.show()

# count1 = len(df[df.Credit_History == 1])
# count0 = len(df[df.Credit_History == 0])
# countNull = len(df[df.Credit_History.isnull()])
# print("Percentage of Good credit history: {:.2f}%".format((count1 / (len(df.Credit_History))*100)))
# print("Percentage of Bad credit history: {:.2f}%".format((count0 / (len(df.Credit_History))*100)))
# print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Credit_History))*100)))

# print(df.Property_Area.value_counts(dropna=False))
# df.Property_Area.value_counts(dropna=False)
# sns.countplot(x="Property_Area", data=df, palette="cubehelix")
# plt.show()

# countUrban = len(df[df.Property_Area == 'Urban'])
# countRural = len(df[df.Property_Area == 'Rural'])
# countSemiurban = len(df[df.Property_Area == 'Semiurban'])
# countNull = len(df[df.Property_Area.isnull()])
# print("Percentage of Urban: {:.2f}%".format((countUrban / (len(df.Property_Area))*100)))
# print("Percentage of Rural: {:.2f}%".format((countRural / (len(df.Property_Area))*100)))
# print("Percentage of Semiurban: {:.2f}%".format((countSemiurban / (len(df.Property_Area))*100)))
# print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Property_Area))*100)))

# print(df.Loan_Status.value_counts(dropna=False))
# df.Loan_Status.value_counts(dropna=False)
# sns.countplot(x="Loan_Status", data=df, palette="YlOrBr")
# plt.show()

# countY = len(df[df.Loan_Status == 'Y'])
# countN = len(df[df.Loan_Status == 'N'])
# countNull = len(df[df.Loan_Status.isnull()])
# print("Percentage of Approved: {:.2f}%".format((countY / (len(df.Loan_Status))*100)))
# print("Percentage of Rejected: {:.2f}%".format((countN / (len(df.Loan_Status))*100)))
# print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Loan_Status))*100)))

# print(df.Loan_Amount_Term.value_counts(dropna=False))
# df.Loan_Amount_Term.value_counts(dropna=False)
# sns.countplot(x="Loan_Amount_Term", data=df, palette="rocket")
# plt.show()

# count12 = len(df[df.Loan_Amount_Term == 12.0])
# count36 = len(df[df.Loan_Amount_Term == 36.0])
# count60 = len(df[df.Loan_Amount_Term == 60.0])
# count84 = len(df[df.Loan_Amount_Term == 84.0])
# count120 = len(df[df.Loan_Amount_Term == 120.0])
# count180 = len(df[df.Loan_Amount_Term == 180.0])
# count240 = len(df[df.Loan_Amount_Term == 240.0])
# count300 = len(df[df.Loan_Amount_Term == 300.0])
# count360 = len(df[df.Loan_Amount_Term == 360.0])
# count480 = len(df[df.Loan_Amount_Term == 480.0])
# countNull = len(df[df.Loan_Amount_Term.isnull()])
#
# print("Percentage of 12: {:.2f}%".format((count12 / (len(df.Loan_Amount_Term))*100)))
# print("Percentage of 36: {:.2f}%".format((count36 / (len(df.Loan_Amount_Term))*100)))
# print("Percentage of 60: {:.2f}%".format((count60 / (len(df.Loan_Amount_Term))*100)))
# print("Percentage of 84: {:.2f}%".format((count84 / (len(df.Loan_Amount_Term))*100)))
# print("Percentage of 120: {:.2f}%".format((count120 / (len(df.Loan_Amount_Term))*100)))
# print("Percentage of 180: {:.2f}%".format((count180 / (len(df.Loan_Amount_Term))*100)))
# print("Percentage of 240: {:.2f}%".format((count240 / (len(df.Loan_Amount_Term))*100)))
# print("Percentage of 300: {:.2f}%".format((count300 / (len(df.Loan_Amount_Term))*100)))
# print("Percentage of 360: {:.2f}%".format((count360 / (len(df.Loan_Amount_Term))*100)))
# print("Percentage of 480: {:.2f}%".format((count480 / (len(df.Loan_Amount_Term))*100)))
# print("Missing values percentage: {:.2f}%".format((countNull / (len(df.Loan_Amount_Term))*100)))

#This section will show mean, count, std, min, max and others using describe function.
#print(df[['ApplicantIncome','CoapplicantIncome','LoanAmount']].describe())

# histogram
# The distribution of Applicant income, Co Applicant Income, and Loan Amount are positively skewed and it has outliers
# sns.set(style="darkgrid")
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))
#
# sns.histplot(data=df, x="ApplicantIncome", kde=True, ax=axs[0, 0], color='green')
# sns.histplot(data=df, x="CoapplicantIncome", kde=True, ax=axs[0, 1], color='skyblue')
# sns.histplot(data=df, x="LoanAmount", kde=True, ax=axs[1, 0], color='orange');
# plt.show()

# print(df.head())
# print(df.dtypes)
# numeric_df = df.select_dtypes(include=[np.number])
# print(numeric_df.head())
# print(numeric_df.dtypes)
# # only numeric values are used for heatmap so below are the values.
# # ApplicantIncome        int64
# # CoapplicantIncome    float64
# # LoanAmount           float64
# # Loan_Amount_Term     float64
# # Credit_History       float64
# numeric_df = numeric_df.fillna(0)
# # Generate the heatmap
# plt.figure(figsize=(10, 7))
# sns.heatmap(numeric_df.corr(), annot=True, cmap='inferno')
# plt.show()

# # Categorical - gender vs married
# # Most male applicants are already married compared to female applicants.
# # Also, the number of not married male applicants are higher compare to female applicants that had not married.
# pd.crosstab(df.Gender,df.Married).plot(kind="bar", stacked=True, figsize=(5,5), color=['#f64f59','#12c2e9'])
# plt.title('Gender vs Married')
# plt.xlabel('Gender')
# plt.ylabel('Frequency')
# plt.xticks(rotation=0)
# plt.show()

# # # Categorical - self employed vs credit history
# # # Most not self employed applicants have good credit compared to self employed applicants.
# pd.crosstab(df.Self_Employed,df.Credit_History).plot(kind="bar", stacked=True, figsize=(5,5), color=['#544a7d','#ffd452'])
# plt.title('Self Employed vs Credit History')
# plt.xlabel('Self Employed')
# plt.ylabel('Frequency')
# plt.legend(["Bad Credit", "Good Credit"])
# plt.xticks(rotation=0)
# plt.show()

# # SVM
# # Select the features and the target
# # Ensure all selected features are numeric and handle any missing values
# features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
# X = df[features].fillna(0)
#
# # Convert categorical target variable to numerical
# y = df['Loan_Status'].map({'Y': 1, 'N': 0})
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
#
# SVCclassifier = SVC(kernel='rbf', max_iter=500)
# SVCclassifier.fit(X_train, y_train)
#
# y_pred = SVCclassifier.predict(X_test)
#
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
#
# from sklearn.metrics import accuracy_score
# SVCAcc = accuracy_score(y_pred,y_test)
# print('SVC accuracy: {:.2f}%'.format(SVCAcc*100))
# # 77.24% at random_state = 3


# # Decision Tree
# features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
# X = df[features].fillna(0)
# #
# # # Convert categorical target variable to numerical
# y = df['Loan_Status'].map({'Y': 1, 'N': 0})
# #
# # # Split the data into training and testing sets
# # # 77.24% at random_state = 3
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
# scoreListDT = []
# for i in range(2, 21):
#     DTclassifier = DecisionTreeClassifier(max_leaf_nodes=i)
#     DTclassifier.fit(X_train, y_train)
#     scoreListDT.append(DTclassifier.score(X_test, y_test))
#
# plt.plot(range(2, 21), scoreListDT)
# plt.xticks(np.arange(2, 21, 1))
# plt.xlabel("Leaf")
# plt.ylabel("Score")
# plt.show()
# DTAcc = max(scoreListDT)
# print("Decision Tree Accuracy: {:.2f}%".format(DTAcc * 100))
# # Decision Tree Accuracy: 80.49%
# # graph is down slope

# # Random forest
# features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
# X = df[features].fillna(0)
# y = df['Loan_Status'].map({'Y': 1, 'N': 0})
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
# scoreListRF = []
# for i in range(2, 25):
#     RFclassifier = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=i)
#     RFclassifier.fit(X_train, y_train)
#     scoreListRF.append(RFclassifier.score(X_test, y_test))
#
# plt.plot(range(2, 25), scoreListRF)
# plt.xticks(np.arange(2, 25, 1))
# plt.xlabel("RF Value")
# plt.ylabel("Score")
# plt.show()
# RFAcc = max(scoreListRF)
# print("Random Forest Accuracy:  {:.2f}%".format(RFAcc * 100))
# #Random Forest Accuracy:  80.49%
# #graph is top then down slope




