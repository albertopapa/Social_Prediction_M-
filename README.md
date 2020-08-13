# Social media for knowledge sharing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

Data_COMP=pd.read_csv('/Users/bruger/Desktop/Social_Media_Preferences (COMP1).csv')
Data_COMP.head()



Data_COMP.columns=['Confusion','Personality','Conflict','Complicated','Accurate','Ambiguous','Large','Transfer File','Numeric','Feedback','Cues','Language','P.Focus','Media Richness','Media Preference']
print('Shape of the dataset: ' + str(Data_COMP.shape))
print(Data_COMP.head())

# features and target
Target = 'Media Preference'
Features = ['Feedback','Cues','Language','P.Focus']

X = Data_COMP[Features]
y = Data_COMP[Target]

X

y

# model 
model = RandomForestClassifier()
model.fit(X, y)
model.score(X, y)

import pickle
pickle.dump(model, open('model.pkl', 'wb'))

data = {  'Feedback': 1
             , 'Cues': 2
             , 'Language': 3
             , 'P.Focus': 4}

data.update((x, [y]) for x, y in data.items())

data

data_df = pd.DataFrame.from_dict(data)
data_df

result = model.predict(data_df)

rtype=type(result)
result

rtype

str(result[0])

# send back to browser
output = {'results': str(result[0])}

output

