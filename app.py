import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

def main():
    html_temp = """
    <div style="background-color:lightblue;padding:16px">
    <h2 style="color:black";text-align:center> Admission Prediction Using ML</h2>
    </div>
    
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    
    model = joblib.load('admission_model')
    
    p1=st.number_input("Enter Your GRE Score")
    p2=st.number_input("Enter Your TOEFL Score")
    p3 = st.slider("Enter University Rating", 1, 5, step=1)
    p4 = st.slider("Enter SOP", 0.0, 5.0, step=0.5)
    p5 = st.slider("Enter LOR", 0.0, 5.0, step=0.5)
    p6 = st.slider("Enter CGPA", 0.0, 10.0, step=0.1)
    p7 = st.slider("Enter Research Experience", 0, 1, step=1)

    if st.button('Predict'):
        input_data = np.array([[p1, p2, p3, p4, p5, p6, p7]])
        input_data_scaled = sc.fit_transform(input_data)
        pred = model.predict(input_data_scaled)
        
        st.balloons()
        if pred == 1:
            st.write("<h3 style='color:green;'>High chance of Getting Admission</h3>", unsafe_allow_html=True)
        else:
            st.write("<h3 style='color:yellow;'>Low chance of Getting Admission</h3>", unsafe_allow_html=True)
    
        st.success('Thank You')
    
if __name__ == '__main__':
    main()