from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

from sklearn import datasets
iris=datasets.load_iris()
iris_data=iris.data
iris_labels=iris.target

x_train,x_test,y_train,y_test=train_test_split(iris_data,iris_labels,test_size=0.20)
# Prints Label no. and their names
for i in range(len(iris.target_names)):
 print("Label", i , "-",str(iris.target_names[i]))
 
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

# Display the results
print("Results of Classification using K-nn with K=1 ")
for r in range(0,len(x_test)):
 print(" Sample:", str(x_test[r]), " Actual-label:", str(y_test[r]), " Predicted-label:",str(y_pred[r]))
print("Classification Accuracy :" , classifier.score(x_test,y_test));

print('Confusion Matrix is as follows')
print(confusion_matrix(y_test,y_pred))
print('Accuracy Metrics')
print(classification_report(y_test,y_pred))
