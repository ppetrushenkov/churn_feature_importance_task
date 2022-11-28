from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from numpy import number
import pandas as pd


def clean_and_prepare_data(data: pd.DataFrame):
    """
    Function prepares the data. It removes needless 'CustomerId' column,
    transform 'TotalCharges' feature to float and sets the 'SeniorCitizen' 
    feature as category.
    
    Returns features as the first variable, and the target column 'Churn' as the second one.
    """
    # Churn to boolean
    data['Churn'].replace({'No': False, 'Yes': True}, inplace=True)

    # TotalCharges to float
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # All objects to categorical type
    data['SeniorCitizen'] = data['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})
    data[data.select_dtypes(['object']).columns] = data.select_dtypes(['object'])\
                                                        .apply(lambda x: x.astype('category'))

    # Drop needless column
    data.drop('customerID', axis=1, inplace=True)

    # Form X and y data
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    
    return X, y


def get_preprocessing_transformer():
    """
    Return Column transformer, that handle numerical and categorical features separately.

    Numerical transform applies Simple imputer with 'mean' strategy, then applies MinMax scaling. 
    Categorical transform applies also Simple imputer with 'most_frequent' strategy, then OneHot encode features.
    """
    numerical_columns = make_column_selector(dtype_include=number)  # Pick only numerical columns
    categorical_columns = make_column_selector(dtype_include='category')  # Pick only categorical columns
    
    # Define preprocess for numerical features
    numerical_transformer = make_pipeline(
        SimpleImputer(strategy='mean'), 
        MinMaxScaler()
    )
    # Define preprocess for categorical features
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(sparse=False, handle_unknown='ignore')
    ) 

    return ColumnTransformer([
        ('numerical_transform', numerical_transformer, numerical_columns),
        ('categorical_transform', categorical_transformer, categorical_columns)
    ])
