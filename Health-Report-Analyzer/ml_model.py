import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def Preprocessiong(dataset):
    Pregnancies_Avg = int(dataset['Pregnancies'].mean())
    dataset['Pregnancies'].fillna(Pregnancies_Avg, inplace=True)

    Glucose_Avg = int(dataset['Glucose'].mean())
    dataset['Glucose'].fillna(Glucose_Avg, inplace=True)

    BloodPressure_Avg = int(dataset['BloodPressure'].mean())
    dataset['BloodPressure'].fillna(BloodPressure_Avg, inplace=True)

    SkinThickness_Avg = int(dataset['SkinThickness'].mean())
    dataset['SkinThickness'].fillna(SkinThickness_Avg, inplace=True)

    Insulin_Avg = int(dataset['Insulin'].mean())
    dataset['Insulin'].fillna(Insulin_Avg, inplace=True)

    BMI_Avg = dataset['BMI'].mean()
    dataset['BMI'].fillna(BMI_Avg, inplace=True)

    DiabetesPedigreeFunction_Avg = dataset['DiabetesPedigreeFunction'].mean()
    dataset['DiabetesPedigreeFunction'].fillna(DiabetesPedigreeFunction_Avg, inplace=True)

    Age_Avg = int(dataset['Age'].mean())
    dataset['Age'].fillna(Age_Avg, inplace=True)
    return dataset

def ml_model(dataset):
    independent_data = dataset.drop(['Outcome'], axis='columns')
    x_train, x_test, y_train, y_test = train_test_split(independent_data, dataset['Outcome'], train_size=0.3, random_state=10)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model
