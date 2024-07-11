##############################################################################################################
# Required Python Packages
##############################################################################################################

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

##############################################################################################################
# File Paths
##############################################################################################################

INPUT_PATH = 'breast-cancer.data'
OUTPUT_PATH = 'breast-cancer.csv'

##############################################################################################################
# Headers
##############################################################################################################

HEADERS = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
           "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin","NormalNucleoli", "Mitoses", "CancerType"]

##############################################################################################################
# Function Name : read_data
# Description   : Read the data into pandas dataframe
# Input         : path of CSV file
# Output        : Gives the data
# Author        : Mahesh Dinkar Pawar
# Date          : 09/07/2024
##############################################################################################################

def read_data(path):
    data = pd.read_csv(path)
    return data

##############################################################################################################
# Function Name : get_headers
# Description   : dataset headers
# Input         : dataset
# Output        : returns the header
# Author        : Mahesh Dinkar Pawar
# Date          : 09/07/2024
##############################################################################################################

def get_headers(dataset):
    return dataset.columns.values

##############################################################################################################
# Function Name : add_headers
# Description   : add the headers to the dataset
# Input         : dataset
# Output        : Updated dataset
# Author        : Mahesh Dinkar Pawar
# Date          : 09/07/2024
##############################################################################################################

def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset

##############################################################################################################
# Function Name : data_file_to_csv
# Input         : nothing
# Output        : Write the data to CSV
# Author        : Mahesh Dinkar Pawar
# Date          : 09/07/2024
##############################################################################################################

def data_file_to_csv():
    # Headers
    headers = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
           "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin","NormalNucleoli", "Mitoses", "CancerType"]
    # Load the dataset into pandas data frame
    dataset = read_data(INPUT_PATH)
    # Add headers to the loaded data
    dataset = add_headers(dataset, headers)
    # Save the loaded dataset into csv format
    dataset.to_csv(OUTPUT_PATH, index = False)
    print("File Saved.......!")

##############################################################################################################
# Function Name : split_dataset
# Description   : split the dataset with train percentage
# Input         : Dataset with related information
# Output        : Dataset after spliting
# Author        : Mahesh Dinkar Pawar
# Date          : 09/07/2024
##############################################################################################################

def spit_dataset(dataset, train_percentage, feature_headers, target_headers):
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_headers],
        train_size = train_percentage)
    return train_x, test_x, train_y, test_y

##############################################################################################################
# Function Name : handel_missing_values
# Description   : filter missing values from the dataset
# Input         : Dataset with missing values
# Output        : Dataset after removing missing values
# Author        : Mahesh Dinkar Pawar
# Date          : 09/07/2024
##############################################################################################################

def handel_missing_values(dataset, missing_values_header, missing_label):
    # Split dataset into train and test dataset
    return dataset[dataset[missing_values_header] != missing_label]

##############################################################################################################
# Function Name : random_forest_classifier
# Description   : to train the random forest classifier with features and target data
# Author        : Mahesh Dinkar Pawar
# Date          : 09/07/2024
##############################################################################################################

def random_forest_classifier(features, target):
    classifire = RandomForestClassifier()
    classifire.fit(features, target)
    return classifire

##############################################################################################################
# Function Name : dataset_statistics
# Description   : Basic statistics of the dataset
# Input         : Dataset
# Output        : Description of dataset 
# Author        : Mahesh Dinkar Pawar
# Date          : 09/07/2024
##############################################################################################################

def dataset_statistics(dataset):
    print(dataset.describe())

##############################################################################################################
# Function Name : main
# Description   : main function from where execution starts
# Author        : Mahesh Dinkar Pawar
# Date          : 09/07/2024
##############################################################################################################

def main():
    # Load the csv fileinto pandas dataframe
    dataset = pd.read_csv(OUTPUT_PATH)
    # Get basic statistics of the loaded dataset
    dataset_statistics(dataset)

    # Filter missing values 
    dataset = handel_missing_values(dataset, HEADERS[6],'?')
    train_x, test_x, train_y, test_y = spit_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])
    
    # Train and Test dataset size details
    print("Train_x Shape ::", train_x.shape)
    print("Train_y Shape ::", train_y.shape)
    print("Test_x Shape ::", test_x.shape)
    print("Test_y Shape ::", test_y.shape)

    # Create randome forest classifier instance
    trained_model = random_forest_classifier(train_x, train_y)
    print("Trained model ::", trained_model)
    predictions = trained_model(test_x)

    for i in range(0, 205):
        print("Actual outcome :: {} and predicted outcome :: {}".format(list(test_y)[i],predictions[i]))

    print("train Accuracy ::", accuracy_score(train_y, trained_model.predict(train_x)))
    print("Test Accuracy ::",accuracy_score(test_y, predictions))
    print("Confusion matrix ::",confusion_matrix(test_y, predictions))

##############################################################################################################
# Application starter
##############################################################################################################

if __name__ == "__main__":
    main()