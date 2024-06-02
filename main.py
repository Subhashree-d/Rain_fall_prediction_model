import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from category_encoders import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib

data = pd.read_csv("data.csv")
data = data.dropna()
X = data.drop(["ANNUAL",'YEAR'],axis=1)
y = data['ANNUAL']
continous_features_used = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']
categorical_features = ['DIVISION']
ct = ColumnTransformer(
                        [("categoricalEncoder",OrdinalEncoder(handle_missing=True), categorical_features ),
                       ('scaler', RobustScaler(), continous_features_used)]
                       ,remainder="passthrough")


model = Pipeline([("columntransform", ct),("clf",RandomForestRegressor(n_estimators=500,max_depth=10000,ccp_alpha=0.001))])
model.fit(X,y)

y_pred = model.predict(X)
print(mean_squared_error(y,y_pred))
print(r2_score(y,y_pred))

joblib.dump(model,"model.sav")