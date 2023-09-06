import streamlit as st
import joblib  # Yeh library trained model ko load karne ke liye istemal hoti hai
import sklearn
import warnings

# Trained model ko load karein
model = joblib.load('LogisticRegression_model.pkl')  # trained_model.pkl aapke model ka file name hai


# Streamlit application ka title set karein
st.title('Machine Learning Model Deployment')


# Input features ko collect karein
feature1 = st.number_input("Enter the Age", min_value=0, max_value=100)
feature2 = st.radio("Select Gender",["Female", "Male"])
if feature2 == "Female":
   feature2=0
else:
   feature2=1

feature3 = st.radio("Select Location",["Houston", "Los Angeles", "Miami", "Chicago", "New York"])
if feature3 == "Houston":
   feature3=0
elif feature3 == "Los Angeles":
   feature3=2
elif feature3 == "Miami":
   feature3=3
elif feature3 == "Chicago":
   feature3=4
else:
   feature3=5            
feature4 = st.number_input("Enter the Subscription_Length_Months(4000-4400)", min_value=4000, max_value=4400)
feature5 = st.number_input("Enter the Monthly_Bill")
feature6 = st.number_input("Enter the Total_Usage_GB", min_value=0, max_value=1000)


# Predict button add karein
if st.button('Predict'):
    # Model se prediction banaien
    prediction = model.predict([[feature1, feature2,feature3,feature4,feature5,feature6]])  # Aap apne model ke requirements ke hisab se features provide karein

    # Prediction ko display karein
    st.write(f'Prediction: {prediction[0]}')
