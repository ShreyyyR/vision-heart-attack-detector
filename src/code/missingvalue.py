import pandas as pd

# Load the data into a DataFrame
data = pd.read_csv('labels.csv')
data['Frame'] = data['Frame'].fillna(value='unknown')

# Check for missing values
print(data.isnull().sum())

# Fill missing values with the mean of the column
data['Label'].fillna(data['Label'].mean(), inplace=True)

# Check again for missing values
print(data.isnull().sum())

