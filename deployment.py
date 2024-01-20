import numpy as np
import joblib
import pickle
import streamlit as st

background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://media.istockphoto.com/id/1386923592/photo/male-soldier.jpg?s=612x612&w=0&k=20&c=z3Y9Z8DDFQDxAgUeW8ZkWGzSgD_V1_oOInJnMLDNNgg=");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

# Load the trained model
fite = joblib.load('Trained_model.sav', 'rb')
loaded_model = fite.fit()
# Reverse mapping dictionaries
attack_type_reverse_mapping = {
    'Shooting': 0,
    'Bombing': 1,
    'Hijacking': 2,
    'Arson': 3,
    'Kidnapping': 4,
    'Assassination': 5
}


perpetrator_reverse_mapping = {
    'Group C': 0,
    'Group A': 1,
    'Group D': 2,
    'Group B': 3
}

weapon_used_reverse_mapping = {
    'Blade Weapons': 0,
    'chemical': 1,
    'explosives': 2,
    'firearms': 3,
    'melee': 4,
}


def map_to_encoded_values(value, reverse_mapping):
    return reverse_mapping.get(value, -1)

def terror_prediction(input_data):

    Attack_Type_encoded = map_to_encoded_values(input_data['Attack_Type'], attack_type_reverse_mapping)
    Perpetrator_encoded = map_to_encoded_values(input_data['Perpetrator'], perpetrator_reverse_mapping)
    Weapon_Used_encoded = map_to_encoded_values(input_data['Weapon_Used'], weapon_used_reverse_mapping)

    Victims_Injured = input_data['Victims_Injured']
    Victims_Deceased = input_data['Victims_Deceased']

    input_data_reshaped = np.array([
        Attack_Type_encoded,
        Perpetrator_encoded,
        Victims_Injured,
        Victims_Deceased,
        Weapon_Used_encoded
    ]).reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'Attack is minor'
    else:
        return 'Attack is major'

def main():

    st.title('Terror Attack Prediction Web App')

    # Input fields
    Attack_Type = st.selectbox('Attack Type', list(attack_type_reverse_mapping.keys()))
    Perpetrator = st.selectbox('Perpetrator', list(perpetrator_reverse_mapping.keys()))
    Victims_Injured = st.number_input('Victims Injured', value=0)
    Victims_Deceased = st.number_input('Victims Deceased', value=0)
    Weapon_Used = st.selectbox('Weapon Used', list(weapon_used_reverse_mapping.keys()))
    prediction = ''

    if st.button('Predict'):
        input_data = {
            'Attack_Type': Attack_Type,
            'Perpetrator': Perpetrator,
            'Victims_Injured': Victims_Injured,
            'Victims_Deceased': Victims_Deceased,
            'Weapon_Used': Weapon_Used
        }
        prediction = terror_prediction(input_data)

    st.success(prediction)

if __name__ == '__main__':
    main()