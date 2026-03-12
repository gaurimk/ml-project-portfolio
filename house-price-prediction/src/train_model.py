import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("data/train.csv")

df = df[['GrLivArea','BedroomAbvGr','FullBath','SalePrice']]
df.dropna(inplace=True)

X = df.drop("SalePrice",axis=1)
y = df["SalePrice"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = LinearRegression()

model.fit(X_train,y_train)

joblib.dump(model,"models/house_price_model.pkl")

print("Model saved!")