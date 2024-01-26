import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
import seaborn as sns
#importing datasets
data_set= pd.read_csv('rice_dataset.csv')
#df=pd.DataFrame(data_set)
print("Actual Dataset")
#print(data_set.tail(10).to_string())
print(data_set.columns)
#print(data_set.describe())
print(data_set.dtypes)
print(data_set.shape)
#print(data_set.isna().sum())
value_ = data_set['Class'].value_counts()
print(value_.to_string())
x = data_set.drop(['Class','id'], axis=1)
y = data_set['Class']
# Splitting the dataset into train test split
from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
#Fitting Decision Tree classifier to the training set
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")
classifier.fit(x_train, y_train)
#Predicting the test set result
y_pred= classifier.predict(x_test)
print("------------PREDICTION----------")
df2=pd.DataFrame({"Actual Result-Y":y_test,"Prediction Result":y_pred})
print(df2.to_string())
##predictions
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import accuracy_score
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))
#confusion matrix0
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test,y_pred, labels=[0,1])
print("CONFUSION MATRICS OF 0 AND 1:", cf)
sns.heatmap(cf, annot=True)
mtp.show()
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#FEATURE importance
feature_names = x.columns
print(feature_names)
feature_importace = pd.DataFrame(classifier.feature_importances_, index= feature_names).sort_values(0, ascending=False)
#print(feature_importace)
sns.barplot(data=feature_importace, x=feature_importace.index, y=0)
# Add title
mtp.xticks(rotation=45, ha='right', fontsize=8)
#plt.figure(figsize=(8, 6))
mtp.title('feature_importace')
mtp.show()
