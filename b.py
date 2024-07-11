import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# File Paths
INPUT_PATH = 'breast-cancer.data'
OUTPUT_PATH = 'breast-cancer.csv'

# Headers
HEADERS = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
           "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "CancerType"]

# Function to read data
def read_data(path):
    data = pd.read_csv(path, header=None)
    return data

# Function to get headers
def get_headers(dataset):
    return dataset.columns.values

# Function to add headers
def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset

# Function to convert data file to CSV
def data_file_to_csv():
    dataset = read_data(INPUT_PATH)
    dataset = add_headers(dataset, HEADERS)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print("File Saved.......!")

# Function to split dataset
def split_dataset(dataset, train_percentage, feature_headers, target_header):
    train_x, test_x, train_y, test_y = train_test_split(
        dataset[feature_headers], dataset[target_header], train_size=train_percentage, random_state=42)
    return train_x, test_x, train_y, test_y

# Function to handle missing values
def handle_missing_values(dataset, missing_values_header, missing_label):
    return dataset[dataset[missing_values_header] != missing_label]

# Function to create random forest classifier
def random_forest_classifier(features, target):
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(features, target)
    return classifier

# Function to get dataset statistics
def dataset_statistics(dataset):
    print(dataset.describe())

# Main function
def main():
    # Convert the data file to CSV
    data_file_to_csv()
    
    # Load the CSV file into pandas dataframe
    dataset = pd.read_csv(OUTPUT_PATH)
    
    # Get basic statistics of the loaded dataset
    dataset_statistics(dataset)
    
    # Filter missing values
    dataset = handle_missing_values(dataset, "BareNuclei", '?')
    dataset["BareNuclei"] = pd.to_numeric(dataset["BareNuclei"])
    
    feature_headers = HEADERS[1:-1]
    target_header = HEADERS[-1]
    
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, feature_headers, target_header)
    
    # Train and Test dataset size details
    print("Train_x Shape ::", train_x.shape)
    print("Train_y Shape ::", train_y.shape)
    print("Test_x Shape ::", test_x.shape)
    print("Test_y Shape ::", test_y.shape)
    
    # Create random forest classifier instance
    trained_model = random_forest_classifier(train_x, train_y)
    print("Trained model ::", trained_model)
    
    predictions = trained_model.predict(test_x)
    
    for i in range(len(test_y)):
        print(f"Actual outcome :: {test_y.values[i]} and predicted outcome :: {predictions[i]}")
    
    print("Train Accuracy ::", accuracy_score(train_y, trained_model.predict(train_x)))
    print("Test Accuracy ::", accuracy_score(test_y, predictions))
    print("Confusion Matrix ::\n", confusion_matrix(test_y, predictions))

# Application starter
if __name__ == "__main__":
    main()
