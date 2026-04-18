import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('mushrooms.csv')

imp_cols = ['odor', 'cap-shape', 'gill-color', 'veil-color', 'spore-print-color', 'cap-color', 'gill-size']
X = pd.get_dummies(df[imp_cols])  #converted alphabets to numbers
y = df['class']

y = df['class'].map({'e':0, 'p':1})  #converted e and p to 0 and 1

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)   #dataset split into 20%test 80%train

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)  
print(accuracy)






