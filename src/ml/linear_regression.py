import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.config import RANDOM_STATE
from src.ml.preprocess import load_pima

X, _ = load_pima()
data = X.dropna(subset=["Glucose"])          # keep rows where the target is known
y = data["Glucose"]
features = data.drop(columns="Glucose")

X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=RANDOM_STATE)

pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler()),
                 ("model", LinearRegression())])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print("Target: plasma glucose (mg/dL)")
print("R2   :", round(r2_score(y_test, pred), 3))
print("RMSE :", round(float(np.sqrt(mean_squared_error(y_test, pred))), 2))
print("MAE  :", round(mean_absolute_error(y_test, pred), 2))
coefs = dict(zip(features.columns, pipe.named_steps["model"].coef_.round(2)))
print("standardised coefficients:", coefs)