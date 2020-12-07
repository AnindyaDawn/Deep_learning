# =============================================================================
# # Artificial Neural Network
# =============================================================================

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__
import keras
import seaborn as sns

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]
print(X)
print(y)

# Encoding categorical data
X1= pd.get_dummies(X, columns =['Geography','Gender'] , drop_first = True)
X = X1

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# =============================================================================
# #Importing the keras libraries and packages
# #sequential model required to initialize ANN
# #dense model required to build the layers of the ANN
# =============================================================================
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
ann = tf.keras.models.Sequential()

# =============================================================================
# # Adding the input layer and the first hidden layer
# #the units=6 is the no. of independent variables plus the dependent variable..
# #..divided by 2..i.e.11+1=12..12/2=6.
# #the activation function used in the layers is rectifier function.
# #kernel_initialize is the weight updater..choosing uniform will update weights close to zero.
# =============================================================================
ann.add(tf.keras.layers.Dense(units=6,kernel_initializer='uniform', activation='relu'))
from keras.layers import Dropout
ann.add(tf.keras.layers.Dropout(0.1))
# =============================================================================
# # Adding the second hidden layer
# =============================================================================
ann.add(tf.keras.layers.Dense(units=6,kernel_initializer='uniform', activation='relu'))
from keras.layers import Dropout
ann.add(tf.keras.layers.Dropout(0.1))
ann.add(tf.keras.layers.Dense(units=6,kernel_initializer='uniform', activation='relu'))
from keras.layers import Dropout
ann.add(tf.keras.layers.Dropout(0.1))
ann.add(tf.keras.layers.Dense(units=6,kernel_initializer='uniform', activation='relu'))
from keras.layers import Dropout
ann.add(tf.keras.layers.Dropout(0.1))
ann.add(tf.keras.layers.Dense(units=6,kernel_initializer='uniform', activation='relu'))
from keras.layers import Dropout
ann.add(tf.keras.layers.Dropout(0.1))
# =============================================================================
# # Adding the output layer..the activation fn is sigmoid if dependent variable is binary
# #if the dependent variable is not binary i.e. 3 or 4 then use softmax. softmax is also sigmoid fn.
# =============================================================================
ann.add(tf.keras.layers.Dense(units=1,kernel_initializer='uniform', activation='sigmoid'))

# =============================================================================
# # Part 3 - Training the ANN
# 
# # =============================================================================
# # # Compiling the ANN
# # #Compiling means adding stocastic gradient decend(SGD) to the ANN.
# # #Optimizer parameter is the SGD fn and most common SGD is 'adam'.
# # #loss parameter is 'binary_crossentropy' if dependent variable has binary outcome.
# # #loss parameter is 'categorical_crossentropy' if dependent variable has more than 2 outcomes.
# # #In the metrics parameter we choose accuracy criteria to evaluate the model. The accuracy..
# # #..increases gradually as we run the ann. 
# =============================================================================
# =============================================================================
ann.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# =============================================================================
# # Training the ANN on the Training set
# #batch_size is the SGD size..If batch_size=1 then weight is updated after each observation..
# #..and this is known as stocastic gradient decend or its also known as reinforcement learning..
# #..and if batch_size is > 1 then its known as mini batch gradient decend or batch learning like batch_size=32.
# #epochs is a round when the whole training set passes the ANN.
# =============================================================================
ann.fit(X_train, y_train, batch_size = 10, epochs = 120)

# Part 4 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.7)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# =============================================================================
# # Predicting the result of a single observation
# =============================================================================

"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

Solution:
"""

new_pred = ann.predict(sc.transform([[ 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
new_pred = (new_pred > 0.5)
print(new_pred)

# =============================================================================
# #Evaluating the ANN
# #K fold cross validation is used to evaluate the ANN.
# =============================================================================
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifer():
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=6,kernel_initializer='uniform', activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6,kernel_initializer='uniform', activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6,kernel_initializer='uniform', activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6,kernel_initializer='uniform', activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6,kernel_initializer='uniform', activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1,kernel_initializer='uniform', activation='sigmoid'))
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return ann
ann = KerasClassifier(build_fn = build_classifer, batch_size = 1, nb_epoch = 100)
accuracies = cross_val_score(estimator = ann, X = X_train, y = y_train, cv= 10, n_jobs= -1)
mean = accuracies.mean()
variance = accuracies.std()
print (mean)
print (variance)

# =============================================================================
# #improving an ANN by using dropout which drops a few neurons if you get a high variance
# #Add the ann.add line to layers of ANN to drop neurons..p=0.1 means 10% neurons will be dropped.
# =============================================================================
from keras.layers import Dropout
ann.add(tf.keras.layers.Dropout(0.1))

#Tuning an ANN with GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifer(optimizer):
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=6,kernel_initializer='uniform', activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6,kernel_initializer='uniform', activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1,kernel_initializer='uniform', activation='sigmoid'))
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return ann
ann = KerasClassifier(build_fn = build_classifer)
parameters = {'batch_size': [1, 20],
              'nb_epoch': [200, 250],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = ann,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_
print(best_parameters , best_accuracy )




















