import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def preprocess_data(path):

    df = pd.read_csv(path)

    # Select useful features
    df = df[['GrLivArea','BedroomAbvGr','FullBath','SalePrice']]

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    numeric_features = X.columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return preprocessor, X_train, X_test, y_train, y_test