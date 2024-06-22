import pickle
import streamlit as st

# Importing necessary libraries
import pandas as pd





from huggingface_hub import hf_hub_download


# Function to scale user input
def scale_input(user_input):

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    user_input_df = pd.DataFrame([user_input], columns=feature_names)
    scaled_input = scaler.transform(user_input_df)
    return  pd.DataFrame(scaled_input, columns=user_input_df.columns)

#download model from hugging face
model_path = hf_hub_download(repo_id="JemimaA/fifa-regression-ensemble", filename="ensemble_model.pkl")

# Load trained model 
with open(model_path , 'rb') as file:
    model = pickle.load(file)


# Feature names
feature_names = ['value_eur', 'age', 'potential', 'movement_reactions', 'wage_eur']

st.title('Player Rating Prediction App ⚽️')

# User input fields
st.sidebar.header('Player Features')
def user_input_features():
    value_eur = st.sidebar.number_input('Value (EUR)', min_value=0, max_value=int(1e9), value=int(1e6))
    wage_eur = st.sidebar.number_input('Wage (EUR)', min_value=0, max_value=int(1e9), value=int(1e6))
    age = st.sidebar.slider('Age', 16, 40, 25)
    potential = st.sidebar.slider('Potential', 1, 100, 50)
    movement_reactions = st.sidebar.slider('Movement Reactions', 1, 100, 50)
    data = {
        'value_eur': value_eur,
        'wage_eur': wage_eur,
        'age': age,
        'potential': potential,
        'movement_reactions': movement_reactions
    }
    return data

input_data = user_input_features()

# Get predictions from model
st.subheader('Prediction')
scaled_input = scale_input(input_data)
prediction = model.predict(scaled_input)
st.write(f"Predicted Player Rating: {prediction[0]:.1f}")

# Explain model's prediction
if st.button('Explain Prediction'):
    st.write('In this App, we are using a simple model that averages the predictions of 3 different models: Random Forest, Gradient Boosting and XGBoost to predict the player rating.')
    st.write('The model was trained on the FIFA Male Legacy Players dataset, which contains data on players from the popular FIFA video game series. The dataset contains information on player attributes such as age, potential, value, etc. The model was trained to predict the player rating based on these attributes.')
    st.write("This is a demo project and doesn't use any advanced model explanation techniques. Use with caution.")
