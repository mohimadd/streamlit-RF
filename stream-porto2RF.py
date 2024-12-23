import streamlit as st
import pickle
import pandas as pd
import numpy as np

# load model 
RF_model = pickle.load(open('Model_RandomForest.sav','rb'))

# create title 
st.title('Late Product Prediction')

# create form input
# seperate into 2 columns

left,right = st.columns(2)

with left:
    shipment_model = st.number_input('input shipment mode')
    st.caption('1 : Plane, 2 : Ship, 3 : Train')

with right:
    customer_rating = st.number_input('input customer rating')
    st.caption('1 : 1 star, 2 : 2 star, 3 : 3 star, 4 : 4 star, 5 : 5 star')

with left:
    product_prior = st.number_input('input product prior before')
    st.caption('How many product purchased before')

with right:
    discount_offerd = st.number_input('input discount offered')
    st.caption('in %')

with left:
    product_weight = st.number_input('input product weight')
    st.caption('in grams')


# prediction code
prediction_result = ''

# prediction button
if st.button('Product on time prediction'):

    columns_predict = ['mode_of_shipment','customer_rating','prior_purchases','discount_offered','weight_in_gms']

    input_user = [[shipment_model,customer_rating,product_prior,discount_offerd,product_weight]]
    input_user = pd.DataFrame(input_user, columns=columns_predict)

    product_prediction = RF_model.predict(input_user)

    if (product_prediction[0] == 1):
        prediction_result = 'Your product prediction late'
    else:
        prediction_result = 'Your product prediction on-time'
    st.success(prediction_result)
