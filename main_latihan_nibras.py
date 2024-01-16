#import library
import streamlit as st
import pandas as pd
import joblib
#end of Import

#load Data
df = pd.read_csv("Heart Attack.csv")
#st.dataframe(df)
#end of load data

#load model
model = joblib.load("modelgbr.joblib")
#end of load data

tab1,tab2,tab3 = st.tabs(["Home","APP","About"])

with tab1:
    st.write("ini home")

with tab2:
    st.write("halo ini app")
    age = st.number_input("Masukan Umur")
    gender = int(st.selectbox("Pilih gender : ",options=['Laki',"cewek"])=="Laki")
    impulse = st.number_input("Masukan impulse: ")
    pressurehight = st.number_input("Masukan darah atas :")
    pressurelow = st.number_input("Masukan darah bawah :")
    glucose = st.number_input("Masukan glukosa :")
    kcm = st.number_input("Masukan kcm :")
    troponin = st.number_input("Masukan troponin :")

    hasil_prediksi=""

    if st.button("prediksi"):
        hasil_prediksi = model.predict([[age,gender,impulse,pressurehight,pressurelow,glucose,kcm,troponin]])
        if hasil_prediksi==1:
            st.write("Positif")
        else:
            st.write("Negative")


with tab3:
    st.write("ini About")
