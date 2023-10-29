import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#loading data
test_data=pd.read_csv("test.csv")
train_data=pd.read_csv("train.csv")

features=[  'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age',
       'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X=train_data[features]
y=train_data.Transported

#Data Processing
X = pd.get_dummies(X, columns=['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP'], drop_first=True)
X = X.fillna(X.mean())  # Fill missing values with the mean

#Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating models
rfc_model = RandomForestClassifier(n_estimators=15, random_state=11)
lr_model = LogisticRegression(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
svm_model = SVC(random_state=42)
knn_model = KNeighborsClassifier()
nb_model = GaussianNB()

models={'Random Forest Classifier':rfc_model, 'Linear Regression':lr_model, 'Decision Tree Model':dt_model,'SVM Model': svm_model,'KNN Model': knn_model,'naive Bayes Model': nb_model}

#model validation
for name, model in models.items():
  model.fit(X_train, y_train)
  accuracy=accuracy_score(y_test, model.predict(X_test))
  print(f"Accuracy of the {name} is {accuracy:.2f}")
