import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
data = pd.read_csv('salarydata.csv')
#Replacing missing '?' marks with mode of the attributes
data['workclass']=data['workclass'].replace('?','Private')
data['occupation']=data['occupation'].replace('?','Prof-specialty')
data['native-country']=data['native-country'].replace('?','United-States')
#Feature engineering
data=data.drop('education',axis=1)
#Label encoding
label_encoder = preprocessing.LabelEncoder()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for i in ['workclass', 'education-num', 'marital-status',
       'occupation', 'relationship', 'race', 'sex','native-country', 'salary']:
    data[i] = label_encoder.fit_transform(data[i])

X = data.drop(['salary'], axis = 1)
y = data['salary']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

#Scaling the model
standardisation = preprocessing.StandardScaler()
X = standardisation.fit_transform(X)
X = pd.DataFrame(X)

from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
m=gb.fit(X_train,y_train.values.ravel())
predictions=gb.predict(X_test)

pickle.dump(m, open('model.pkl', 'wb'))

