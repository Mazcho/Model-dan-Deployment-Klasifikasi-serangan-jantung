#Import Library
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier


#call Model
model = pickle.load(open('modelgbr.pkl', 'rb'))

st.header("Prediksi Potensi terkena serangan jantung")
st.write('Silahkan masukkan hasil dari pengecekan dari dokter kedalam kolom dibawah ini untuk mengecek')
col1,col2 = st.columns(2)
with col1 :
    #membuat inputan umur
    age = st.number_input("Masukan umur: ", value=0, key="age")
with col1:
    #membuat gender option
    gender_option={
        1 : "Laki-Laki",
        2 : "Perempuan"
    }
gender = st.selectbox("Pilih Jenis Kelamin",options=list(gender_option.keys()), format_func=lambda x: gender_option[x])
with col1:
    glucose = st.number_input("Masukan Kadar Glukosa : ",value=0, key="glucose")
    kcm = st.number_input("Masukan Kreatin Serum (KCM) : ",value=0.0, step = 0.1, key="kcm")
with col2:
    impulse = st.number_input("Masukan impulse: ", value=0, key="impluse")
    pressurehight = st.number_input("Masukan Tekanan darah bagian atas : ",value=0, key="pressurehight")
    pressurelow = st.number_input("Masukan Tekanan darah bagian bawah : ",value=0, key="pressurelow")
    troponin = st.number_input("Masukan troponin : ",min_value=0.00, step = 0.001, key="troponin")

    #prediksi ML
    prediksi_serangan_jantung = ''

    #tombol prediksi
if st.button("Prediksi"):
    prediksi_serangan_jantung = model.predict([[age,gender,impulse,pressurehight,pressurelow,glucose,kcm,troponin]])
    if prediksi_serangan_jantung == 1:
        col4,col5 = st.columns(2)
        with col4:
            st.image("heartatt.png",width=400)
        with col5:
            st.write("Hasil")
            if pressurehight <=139 and pressurelow<90:
                st.write("Tekanan darah Normal")
            elif pressurehight<=160 and pressurelow<=90:
                st.write("Anda memiliki hipertensi ringan")
            elif pressurehight<=180 and pressurelow<=110:
                st.write("Anda memiliki hipertensi sedang")
            elif pressurehight<=200 and pressurelow<=120:
                st.write("Anda memiliki hipertensi berat")
            elif pressurehight>200 and pressurelow>120:
                st.write("Anda memiliki hipertensi Maligna")
            if glucose >100:
                st.write("glucose darah tidak normal (Prediabetes)")
            elif glucose>125:
                st.write("glucose darah tinggi (Diabetes)")  
            st.error('Memiliki Resiko Serangan Jantung')
    else:
        col7,col8 = st.columns(2)
        with col7:
             st.image("jantung.png",width=300)
        with col8:
            st.write("Hasil")
            if pressurehight <=139 and pressurelow<90:
                st.write("Tekanan darah Normal")
            elif pressurehight<=160 and pressurelow<=90:
                st.write("Anda memiliki hipertensi ringan")
            elif pressurehight<=180 and pressurelow<=110:
                st.write("Anda memiliki hipertensi sedang")
            elif pressurehight<=200 and pressurelow<=120:
                st.write("Anda memiliki hipertensi berat")
            elif pressurehight>200 and pressurelow>120:
                st.write("Anda memiliki hipertensi Maligna")
            if glucose >100:
                st.write("glucose darah tidak normal (Prediabetes)")
                st.write('Sebaiknya perhatikan konsumsi glucose anda agar tetap mejaga kesehatan jantung dengan baik')
            elif glucose>125:
                st.write("glucose darah tinggi (Diabetes)")
            st.success('Tidak Memiliki Resiko Serangan Jantung')
