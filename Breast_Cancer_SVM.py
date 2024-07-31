##############################################################################################################
# Breast Cancer Dataset with Support Vector Machine
# Required Python Packages
##############################################################################################################

from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm

##############################################################################################################
# Function Name : Super_Vector
# Description   : all operation covered
# Output        : Gives the accuracy
# Author        : Mahesh Dinkar Pawar
# Date          : 30/07/2024
##############################################################################################################

def Super_Vector():
    Cancer = datasets.load_breast_cancer()

    print("Features of the cancer dataset : ",Cancer.feature_names)
    print("Labels of cancer dataset : ",Cancer.target_names)

    print("Shape of the dataset is : ",Cancer.data.shape)
    print("First 5 record are : ")
    print(Cancer.data[0:5])

    print("Target of dataset :",Cancer.target)

    X_train, X_test, Y_train, Y_test = train_test_split(Cancer.data, Cancer.target, test_size=0.30, random_state=109)

    clf = svm.SVC(kernel='linear')

    clf.fit(X_train, Y_train)

    Y_Pred = clf.predict(X_test)

    print("Accuracy of the model is : ",metrics.accuracy_score(Y_test, Y_Pred)*100,"%")

def main():
    print("-------------------Mahesh Pawar-------------------")
    Super_Vector()

##############################################################################################################
# Application starter
##############################################################################################################

if __name__ == "__main__":
    main()