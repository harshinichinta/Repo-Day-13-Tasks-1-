#----task 1----
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("Machine Learning App - Train Dataset")

# 1️⃣ Load Dataset
st.header("1. Load Dataset")
df = pd.read_csv("train.csv")
st.write(df.head())

# 2️⃣ Understand Data & Cleaning
st.header("2. Data Understanding")
st.write("Dataset Shape:", df.shape)
st.write("Missing Values:")
st.write(df.isnull().sum())

# 3️⃣ Input and Output Variables
st.header("3. Feature Selection")

columns = df.columns.tolist()
target = st.selectbox("Select Target Variable", columns)

features = st.multiselect("Select Feature Variables", columns)

if features and target:

    X = df[features]
    y = df[target]

    # Visualization
    st.subheader("Data Visualization")
    fig, ax = plt.subplots()
    ax.scatter(df[features[0]], df[target])
    st.pyplot(fig)

    # 4️⃣ Train-Test Split
    st.header("4. Train Test Split")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.write("Training Data Size:", X_train.shape)
    st.write("Testing Data Size:", X_test.shape)

    # 5️⃣ Create Model & Train
    st.header("5. Model Training")

    model = LinearRegression()
    model.fit(X_train, y_train)

    st.success("Model trained successfully!")

    # 6️⃣ Testing
    st.header("6. Model Testing")

    predictions = model.predict(X_test)

    score = r2_score(y_test, predictions)

    st.write("R2 Score:", score)

    # 7️⃣ Prediction
    st.header("7. Prediction")

    input_data = []

    for feature in features:
        val = st.number_input(f"Enter {feature}")
        input_data.append(val)

    if st.button("Predict"):

        result = model.predict([input_data])

        st.success(f"Prediction Result: {result[0]}")




        #task 2---------
        import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("Insurance Cost Prediction App")

# 1️⃣ Load Dataset
st.header("1. Load Dataset")

df = pd.read_csv("insurance.csv")
st.write(df.head())


# 2️⃣ Understand Data & Cleaning
st.header("2. Data Understanding")

st.write("Dataset Shape:", df.shape)
st.write("Missing Values:")
st.write(df.isnull().sum())


# Convert categorical variables
df = pd.get_dummies(df, drop_first=True)


# 3️⃣ Input and Output Variables
st.header("3. Feature Selection")

X = df.drop("charges", axis=1)
y = df["charges"]

st.write("Feature Variables:", X.columns)


# 4️⃣ Train-Test Split
st.header("4. Train Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write("Training Data Size:", X_train.shape)
st.write("Testing Data Size:", X_test.shape)


# 5️⃣ Create Model & Train
st.header("5. Model Training")

model = LinearRegression()
model.fit(X_train, y_train)

st.success("Model trained successfully")


# 6️⃣ Testing
st.header("6. Model Testing")

predictions = model.predict(X_test)

score = r2_score(y_test, predictions)

st.write("Model R2 Score:", score)


# 7️⃣ Prediction
st.header("7. Insurance Cost Prediction")

age = st.number_input("Enter Age")
bmi = st.number_input("Enter BMI")
children = st.number_input("Number of Children")

if st.button("Predict Insurance Cost"):

    input_data = [[age, bmi, children, 0, 0]]

    prediction = model.predict(input_data)

    st.success(f"Predicted Insurance Cost: {prediction[0]}")



    #task 3---------
    import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("NYC Taxi Trip Prediction App")

# 1️⃣ Load Dataset
st.header("1. Load Dataset")

df = pd.read_csv("taxi_tripdata.csv")

st.write(df.head())


# 2️⃣ Understand Data & Cleaning
st.header("2. Data Understanding")

st.write("Dataset Shape:", df.shape)

st.write("Missing Values")
st.write(df.isnull().sum())


# 3️⃣ Input and Output Variables
st.header("3. Feature Selection")

# Example columns (change based on dataset)
X = df[['passenger_count', 'fare_amount']]
y = df['trip_distance']

st.write("Feature Variables:", X.columns)


# Visualization
st.subheader("Visualization")

fig, ax = plt.subplots()

ax.scatter(df['fare_amount'], df['trip_distance'])

st.pyplot(fig)


# 4️⃣ Train Test Split
st.header("4. Train Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write("Training Data:", X_train.shape)

st.write("Testing Data:", X_test.shape)


# 5️⃣ Create Model & Train
st.header("5. Model Training")

model = LinearRegression()

model.fit(X_train, y_train)

st.success("Model Trained Successfully")


# 6️⃣ Testing
st.header("6. Model Testing")

predictions = model.predict(X_test)

score = r2_score(y_test, predictions)

st.write("Model R2 Score:", score)


# 7️⃣ Prediction
st.header("7. Taxi Trip Prediction")

passenger_count = st.number_input("Passenger Count")

fare_amount = st.number_input("Fare Amount")

if st.button("Predict Trip Distance"):

    result = model.predict([[passenger_count, fare_amount]])

    st.success(f"Predicted Trip Distance: {result[0]}")