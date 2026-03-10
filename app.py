import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.title("Machine Learning Tasks Deployment")

task = st.sidebar.selectbox(
    "Select Task",
    ("House Price Prediction", "Insurance Charges Prediction", "Taxi Trip Duration Prediction")
)

# ===============================
# TASK 1 : HOUSE PRICE PREDICTION
# ===============================
if task == "House Price Prediction":

    st.header("House Price Prediction")

    data = pd.read_csv(r"C:\Tekworks\Day 13 - tasks\train (1).csv")

    important_columns = [
        "SalePrice","OverallQual","GrLivArea","GarageCars",
        "TotalBsmtSF","YearBuilt","Neighborhood","LotArea","KitchenQual"
    ]

    data = data[important_columns]

    data["GarageCars"].fillna(data["GarageCars"].median(), inplace=True)
    data["TotalBsmtSF"].fillna(data["TotalBsmtSF"].median(), inplace=True)
    data["KitchenQual"].fillna(data["KitchenQual"].mode()[0], inplace=True)

    data = pd.get_dummies(data, columns=["Neighborhood","KitchenQual"], drop_first=True)

    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    model = LinearRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)

    st.write("RMSE:", rmse)
    st.write("R2 Score:", r2)

# ===============================
# TASK 2 : INSURANCE PREDICTION
# ===============================
elif task == "Insurance Charges Prediction":

    st.header("Insurance Charges Prediction")

    data = pd.read_csv(r"C:\Tekworks\Day 13 - tasks\insurance.csv")

    important_columns = [
        "charges","age","sex","bmi","children","smoker","region"
    ]

    data = data[important_columns]

    def bmi_category(bmi):
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"

    data["bmi_category"] = data["bmi"].apply(bmi_category)

    def age_category(age):
        if age < 18:
            return "Child"
        elif age < 30:
            return "Young Adult"
        elif age < 50:
            return "Adult"
        else:
            return "Senior"

    data["age_category"] = data["age"].apply(age_category)

    data["smoker_bmi"] = data["bmi"] * (data["smoker"] == "yes").astype(int)

    data = pd.get_dummies(data, drop_first=True)

    X = data.drop("charges", axis=1)
    y = data["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    model = LinearRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)

    st.write("MAE:", mae)
    st.write("RMSE:", rmse)
    st.write("R2 Score:", r2)

# ===============================
# TASK 3 : TAXI TRIP PREDICTION
# ===============================
elif task == "Taxi Trip Duration Prediction":

    st.header("Taxi Trip Duration Prediction")

    data = pd.read_csv(r"C:\Tekworks\Day 13 - tasks\NYC.csv")

    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])

    data["hour"] = data["pickup_datetime"].dt.hour
    data["day_of_week"] = data["pickup_datetime"].dt.dayofweek

    data["rush_hour"] = data["hour"].apply(
        lambda x: 1 if (7<=x<=9 or 16<=x<=19) else 0
    )

    from math import radians, cos, sin, asin, sqrt

    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians,[lon1,lat1,lon2,lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r

    data["distance"] = data.apply(
        lambda row: haversine(
            row["pickup_longitude"],
            row["pickup_latitude"],
            row["dropoff_longitude"],
            row["dropoff_latitude"]
        ),axis=1
    )

    data["trip_duration"] = np.log1p(data["trip_duration"])

    important_columns = [
        "trip_duration","passenger_count","distance",
        "hour","day_of_week","rush_hour"
    ]

    data = data[important_columns]

    X = data.drop("trip_duration", axis=1)
    y = data["trip_duration"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    model = LinearRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)

    st.write("RMSE:", rmse)
    st.write("R2 Score:", r2)