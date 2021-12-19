from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

train = pd.read_csv(r'C:\Users\Administrator\Pictures\My Codes\Titanic\train.csv')
test = pd.read_csv(r'C:\Users\Administrator\Pictures\My Codes\Titanic\test.csv')
submission = pd.read_csv(r'C:\Users\Administrator\Pictures\My Codes\Titanic\gender_submission.csv')


# چک وجود داده خالی
print(train.isnull().sum())

# Missing Data به صورت بصری
#def missingdata(data):
#    total = data.isnull().sum().sort_values(ascending = False)
#    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
#    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#    ms= ms[ms["Percent"] > 0]
#    f,ax =plt.subplots(figsize=(8,6))
#    plt.xticks(rotation='90')
#    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)
#    plt.xlabel('Features', fontsize=15)
#    plt.ylabel('Percent of missing values', fontsize=15)
#    plt.title('Percent missing data by feature', fontsize=15)
#    return ms
#missingdata(train)

def normalize2 (x) :
    y = ((x-np.min(x))/(np.max(x)-np.min(x))) + 1
    return y

features = ['Pclass' , 'Sex' , 'SibSp' , 'Parch' ,'Embarked' , 'Age']
train_2 = train[features]
test_2 = test[features]
target = train.Survived

def pre_processing (x) :
    x.Sex = x.Sex.map ({ 'male' : 1 , 'female' : 2 })
    x.Age = x.Age.fillna (np.mean(x.Age))
    x.Age = normalize2 (x.Age)
    x.Pclass = normalize2 (x.Pclass)
    x.SibSp = normalize2 (x.SibSp)
    x.Parch = normalize2 (x.Parch)
    x.Embarked = x.Embarked.map ({ 'S' : 1 , 'C' : 2 , 'Q' : 3})
    x.Embarked = x.Embarked.fillna (4)
#    x. = x..map({ '' : 1 , '' : 2 , '' : 3 , '' : 4 , '' : 5 , '' : 6 , '' : 7 , '' : 8 })


    # روشی دیگر
    #x['Embarked'].fillna(x['Embarked'].mode()[0], inplace = True)
    return x

train_3 = pre_processing (train_2)
test_3 = pre_processing (test_2)

# چک وجود داده خالی
print(train_3.isnull().sum())


model = MLPClassifier (hidden_layer_sizes = ([6,6]) , activation = 'tanh' , solver = 'lbfgs' , max_iter = 250)
model.fit (train_3 , target)

result = model.predict (test_3)
submission['Survived'] = result

submission.to_csv(r'C:\Users\Administrator\Pictures\My Codes\Titanic\3rdkaggle.csv' , index = False)
