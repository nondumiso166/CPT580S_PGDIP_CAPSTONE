import pandas as pd

# This function is for reading the data file into a pd dataframe.

def read_dataset(file_path, **kwargs):
    """Read data from CSV or Excel files"""
    if file_path.endswith('.csv'):
        raw_data = pd.read_csv(file_path, **kwargs)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        raw_data = pd.read_excel(file_path, **kwargs)
    else:
        raise ValueError("Unsupported file format. Use")


# describe dataset datatypes, list total of columns with missing values , and remove nulls
def examine(dataframe):
    print("Types of the variables we have:")
    print(dataframe.dtypes)

    print("Total Samples with missing values:")
    print(dataframe.isnull().any(axis=1).sum())

    print("Total Missing Values per Variable")
    print(dataframe.isnull().sum())


# Function to remove null values
def null_values(df):
    df = df.dropna()
    return df
