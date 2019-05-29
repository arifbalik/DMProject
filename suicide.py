import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

pima = pd.read_csv("foreveralone.csv")

#Mapping values to discrete integers (Normalization)

weight_m = {'Normal weight': 0, 'Underweight': 1, 'Overweight': 2, 'Obese': 3}
pima['bodyweight'] = pima['bodyweight'].map(weight_m)

income_m = {'$30,000 to $39,999': 0, '$1 to $10,000': 1, '$0': 2, '$50,000 to $74,999': 3,
'$20,000 to $29,999': 4, '$10,000 to $19,999': 5, '$75,000 to $99,999': 6,
 '$150,000 to $174,999': 7, '$125,000 to $149,999': 8, '$100,000 to $124,999': 9,
 '$174,999 to $199,999': 10, '$40,000 to $49,999': 11, '$200,000 or more':  12}
pima['income'] = pima['income'].map(income_m)

gender_m = {'Male': 0, 'Female': 1, 'Transgender male':2, 'Transgender female':3}
pima["gender"] = pima["gender"].map(gender_m)

yes_no_m = {'Yes': 1, 'No': 0}
pima["attempt_suicide"] = pima["attempt_suicide"].map(yes_no_m)
pima["depressed"] = pima["depressed"].map(yes_no_m)
pima["social_fear"] = pima["social_fear"].map(yes_no_m)
pima["virgin"] = pima["virgin"].map(yes_no_m)
pima["prostitution_legal"] = pima["prostitution_legal"].map(yes_no_m)

pay_m = {"No": 0, 'Yes': 1, "Yes but I haven't": 2}
pima["pay_for_sex"] = pima["pay_for_sex"].map(pay_m)

#Drop all NaN values
pima.dropna(inplace=True)

#See data info to get some insight
print(pima["income"].describe())
print(pima.describe())

#two different feature, these two will be tested for every model
feature_cols = ['age','friends', 'depressed','social_fear', "virgin", "gender", "pay_for_sex", "income", "bodyweight"]
feature_cols2 = ['friends', 'depressed','social_fear', "virgin", "income", "bodyweight"]

#predicted value
y = pima.attempt_suicide

#Models
MLP = MLPClassifier(solver='adam', learning_rate='adaptive')
DT = DecisionTreeClassifier()
RF = RandomForestClassifier()
KN = KNeighborsClassifier()

performance = []
results = [] #holds 4 experiment for a model
best_exp = [] #holds best experiments for each model

#Now here we have 2 loops. The outer loop runs for times and each time a model will be selected, and after model is
#selected inner loop will run four times and every time it will either change test size or features and will run the
#model. After all different configurations is runned, best result will be found and stored in best_exp.
for I in range(4):
    if (I == 0):
        model = DT
        print("Model: Decision Tree")
    elif (I == 1):
        model = RF
        print("Model: Random Forest")
    elif (I == 2):
        model = KN
        print("Model: K-Neighbors")
    elif (I == 3):
        model = MLP
        print("Model: MLP Neural Network")
    for K in range(4):
        if(K == 0):
            t_size = 0.1
            col = feature_cols
            X = pima[feature_cols]
        elif(K == 1):
            t_size = 0.1
            col = feature_cols2
            X = pima[feature_cols2]
        elif (K == 2):
            t_size = 0.4
            col = feature_cols
            X = pima[feature_cols]
        elif (K == 3):
            t_size = 0.4
            col = feature_cols2
            X = pima[feature_cols2]
        exp_no = (I + 1) + ((K+1) / 10)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=1)
        model = model.fit(X_train, y_train)
        pred = model.predict(X_test)

        print("\nExperiment #",exp_no, ", Accuracy: ", metrics.accuracy_score(y_test, pred))
        print("Test Size: ", t_size * 100 ,"%")
        print("Feature Columns: ", col)
        print(classification_report(y_test, pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
        results.append(metrics.accuracy_score(y_test, pred))

    performance.append(max(results))
    best_exp.append((I + 1) + (results.index(max(results)) +1)/10)
    print("\nBest Experiment: #",best_exp[I],", Accuracy: ", max(results))
    del results[:]


#Plot results
objects = ( 'Decision Tree', 'Random Forest', 'K-Neighbors', 'MLP')
y_pos = np.arange(len(objects))

bars = plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
ix = 0
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval - .05, '#')
    plt.text(bar.get_x() + 0.09, yval - .05, best_exp[ix])
    plt.text(bar.get_x(), yval + .005, '%.2f' % yval)
    ix = ix + 1
plt.ylabel('Accuracy')
plt.title('Model Performances')
plt.show()