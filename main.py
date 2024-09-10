import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
# Load dataset
dataset = pd.read_excel("HousePricePrediction.xlsx")

# Drop the 'Id' column as it's not needed for modeling
dataset.drop(['Id'], axis=1, inplace=True)

#This line fills missing values in target variable
#This may effect the final percentage error ,making some better and making some worse,you may choose if you want to keep it
#dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())

# Drop rows with missing values in any feature column
new_dataset = dataset.dropna()
# Identify categorical columns in the dataset
s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)

# Initialize the OneHotEncoder to convert categorical features to binary vectors
OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Apply OneHotEncoder to the categorical columns and create a DataFrame
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))

# Set index and column names for the encoded DataFrame to match the original dataset
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()

# Drop the original categorical columns and concatenate the encoded columns
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Define features (X) and target variable (Y)
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split the dataset into training and validation sets
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

#Model training
#1:SVM – Support vector Machine
model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
# Make predictions on the validation set
Y_pred = model_SVR.predict(X_valid)
# Print the Mean Absolute Percentage Error of the predictions
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(Y_valid, Y_pred))

#2:Random Forest Regression
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(Y_valid, Y_pred))

#3:Linear Regression
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(Y_valid, Y_pred))

#4:CatBoost Classifier
cb_model = CatBoostRegressor()
cb_model.fit(X_train, Y_train)
preds = cb_model.predict(X_valid)
print("Mean Absolute Percentage Error:",mean_absolute_percentage_error(Y_valid, preds))
