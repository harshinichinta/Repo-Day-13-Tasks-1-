import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ---------------- TASK 1 ----------------
st.title("Task 1 - Machine Learning App")

df = pd.read_csv("train.csv")
st.write(df.head())

st.write("Dataset Shape:", df.shape)
st.write("Missing Values")
st.write(df.isnull().sum())

columns = df.columns.tolist()

target = st.selectbox("Select Target Variable", columns)
features = st.multiselect("Select Feature Variables", columns)

if features and target:

    X = df[features]
    y = df[target]

    fig, ax = plt.subplots()
    ax.scatter(df[features[0]], df[target])
    st.pyplot(fig)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = r2_score(y_test, predictions)

    st.write("R2 Score:", score)

    input_data = []

    for feature in features:
        val = st.number_input(f"Enter {feature}")
        input_data.append(val)

    if st.button("Predict"):
        result = model.predict([input_data])
        st.success(result[0])


# ---------------- TASK 2 ----------------
st.title("Task 2 - Insurance Cost Prediction")

df2 = pd.read_csv("insurance.csv")
st.write(df2.head())

df2 = pd.get_dummies(df2, drop_first=True)

X = df2.drop("charges", axis=1)
y = df2["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model2 = LinearRegression()
model2.fit(X_train, y_train)

predictions = model2.predict(X_test)

score = r2_score(y_test, predictions)

st.write("Model R2 Score:", score)

age = st.number_input("Enter Age")
bmi = st.number_input("Enter BMI")
children = st.number_input("Children")

if st.button("Predict Insurance Cost"):

    input_data = [[age, bmi, children, 0, 0]]

    prediction = model2.predict(input_data)

    st.success(prediction[0])


# ---------------- TASK 3 ----------------
st.title("Task 3 - NYC Taxi Trip Prediction")

df3 = pd.read_csv("taxi_tripdata.csv")

st.write(df3.head())

X = df3[['passenger_count', 'fare_amount']]
y = df3['trip_distance']

fig, ax = plt.subplots()
ax.scatter(df3['fare_amount'], df3['trip_distance'])
st.pyplot(fig)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

model3 = LinearRegression()
model3.fit(X_train, y_train)

predictions = model3.predict(X_test)

score = r2_score(y_test, predictions)

st.write("Model R2 Score:", score)

passenger_count = st.number_input("Passenger Count")
fare_amount = st.number_input("Fare Amount")

if st.button("Predict Trip Distance"):

    result = model3.predict([[passenger_count, fare_amount]])

    st.success(result[0])