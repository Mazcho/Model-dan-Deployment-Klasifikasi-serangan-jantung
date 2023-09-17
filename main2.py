#Import Library
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier


#call Model
model = pickle.load(open('model.pkl',"rb"))

#call data
#call data
file_data = "Heart Attack.csv"
try:
    df = pd.read_csv(file_data)
except FileNotFoundError:
    st.error("File CSV tidak ditemukan. Pastikan file ada dalam direktori yang benar atau ganti nama file sesuai dengan yang Anda miliki.")
    st.stop()

st.title(":red[_HELLO_]  :blue[_HEART_] WEBSITE")
#membuat tab
tab1,tab2,tab3= st.tabs(["Home","App","Tentang Dataset"])

with tab1:
    st.header("Hi, Selamat data di website hello heart")
    col11,col12 = st.columns(2)
    with col11:
        st.image("heart-rate_9108205.png",width = 300)
    with col12:
        st.write("Hello heart adalah sebuah website yang digunakan untuk melakukan pendeteksian apakah seseorang ini berpotensi terkena penyakit jantung Cardiovascular illnesses (CVDs) . Web app ini berawal dari pengambilan sampel dari 1319 orang di wilayah boston pada 2 bulan terahkir. App ini menggunakan model dari Gradient Boosting Regresor dengan tinggal akurasi sebesar 99% untuk masing masing kelasnya. Jadi tidak perlu diragukan lagi untuk menggunakan web app ini untuk mendeteksi 0potensi serangan jantung pada masyarakat sekitar.")
    
    st.header("Analisis Singkat")
    st.write("Mengambil dari dataset yang ada, terdapat banyak orang yang terkena penyakit jantung pada umur 60 hingga 70 tahun adanya.Menurut grafik dibawah, menunjukkan bahwa mulai umur 20 pun juga sudah terkena penyakit serangan antung. Tentu hal ini sangat memprihatinkan untuk para pemuda pemuda di masa sekarang.")
    st.write("")

    #membuat grafik
    tingkatkematian_umur = df.groupby(["age","class"]).size().unstack()
    st.bar_chart(tingkatkematian_umur)

    #grafik 2
    st.write("Dari jumlah kasus kematian yang ada pun juga jauh lebih tinggi dari pada yanh tidak meninggal. Grafik dibaah ini menunjukkan bahwa, 810 dari 1319 orang terkena penyakit jantung")
    perbandingan_mati_hidup = df["class"].value_counts()
    st.bar_chart(perbandingan_mati_hidup)

    st.header("Kesimpulan dari hasil analisis singkat")
    col9,col10 = st.columns(2)
    with col9:
        st.write("Bisa kita lihat pada hasil diagram diatas, menunjukkan bahwa pada kelas negativ ( orang yang tidak memiliki potensi serangan jantung) jumlah lebih sedikit dari pada orang oarang yang terkena potensi serang jantung. Kita ketahui bahwa rasa peduli orang orang dengan kesehatan jantungnya sangat sedikit sekali ")
    with col10:
        st.write("Maka dari itu , banyak orang masih belum mengetahui cara menjaga kesehatan dari jantung. Menjaga kesehatan jantung itu sangatlah penting untuk kesehatan. Maka dari itu website Hello Heart menyajikan App berbasis web untuk melakukan pengecekan terhadap kondisi anda, dan memberikan tips untuk menjaga keseahtan jantung kalian semua")
    

with tab2:
    st.header("Prediksi Potensi terkena serangan jantung")
    st.write('Silahkan masukkan hasil dari pengecekan dari dokter kedalam kolom dibawah ini untuk mengecek')
    col1,col2 = st.columns(2)
    with col1 :
        #membuat inputan umur
        umur = st.number_input("Masukan umur: ", value=0)
    with col1:
        #membuat gender option
        gender_option={
            1 : "Laki-Laki",
            2 : "Perempuan"
        }
        jenis_kelamin = st.selectbox("Pilih Jenis Kelamin",options=list(gender_option.keys()), format_func=lambda x: gender_option[x])
        selected_value = gender_option[jenis_kelamin]
    with col1:
        glukosa = st.number_input("Masukan Kadar Glukosa : ",value=0)
        kcm = st.number_input("Masukan Kreatin Serum (KCM) : ",value=0.0, step = 0.1)
    with col2:
        impulse = st.number_input("Masukan impulse: ", value=0)
        tekanan_atas = st.number_input("Masukan Tekanan darah bagian atas : ",value=0)
        tekanan_bawah = st.number_input("Masukan Tekanan darah bagian bawah : ",value=0)
        troponin = st.number_input("Masukan troponin : ",min_value=0.00, step = 0.001)

    #prediksi ML
    prediksi_serangan_jantung = ''

    #tombol prediksi
    if st.button("Prediksi"):
        prediksi_serangan_jantung = model.predict([[umur,jenis_kelamin,impulse,tekanan_atas,tekanan_bawah,glukosa,kcm,troponin]])
        if prediksi_serangan_jantung == 1:
            col4,col5 = st.columns(2)
            with col4:
                st.image("heartatt.png",width=400)
            with col5:
                st.write("Hasil")
                if tekanan_atas <=139 and tekanan_bawah<90:
                    st.write("Tekanan darah Normal")
                elif tekanan_atas<=160 and tekanan_bawah<=90:
                    st.write("Anda memiliki hipertensi ringan")
                elif tekanan_atas<=180 and tekanan_bawah<=110:
                    st.write("Anda memiliki hipertensi sedang")
                elif tekanan_atas<=200 and tekanan_bawah<=120:
                    st.write("Anda memiliki hipertensi berat")
                elif tekanan_atas>200 and tekanan_bawah>120:
                    st.write("Anda memiliki hipertensi Maligna")
                if glukosa >100:
                    st.write("Glukosa darah tidak normal (Prediabetes)")
                elif glukosa>125:
                    st.write("Glukosa darah tinggi (Diabetes)")  
                st.error('Memiliki Resiko Serangan Jantung')
        else:
            col7,col8 = st.columns(2)
            with col7:
                st.image("jantung.png",width=300)
            with col8:
                st.write("Hasil")
                if tekanan_atas <=139 and tekanan_bawah<90:
                    st.write("Tekanan darah Normal")
                elif tekanan_atas<=160 and tekanan_bawah<=90:
                    st.write("Anda memiliki hipertensi ringan")
                elif tekanan_atas<=180 and tekanan_bawah<=110:
                    st.write("Anda memiliki hipertensi sedang")
                elif tekanan_atas<=200 and tekanan_bawah<=120:
                    st.write("Anda memiliki hipertensi berat")
                elif tekanan_atas>200 and tekanan_bawah>120:
                    st.write("Anda memiliki hipertensi Maligna")
                if glukosa >100:
                    st.write("Glukosa darah tidak normal (Prediabetes)")
                    st.write('Sebaiknya perhatikan konsumsi glukosa anda agar tetap mejaga kesehatan jantung dengan baik')
                elif glukosa>125:
                    st.write("Glukosa darah tinggi (Diabetes)")
                st.success('Tidak Memiliki Resiko Serangan Jantung')

    
with tab3:
    st.header("Tentang Dataset")
    st.image('header.jpg')
    col13,col14 = st.columns(2)
    with col13:
        st.image("dataset.jpg",width=350)
    with col14:
        st.write("Penyakit kardiovaskular (CVD) adalah penyebab utama kematian di seluruh dunia. CVD termasuk penyakit jantung koroner, penyakit serebrovaskular, penyakit jantung rematik, dan masalah jantung dan pembuluh darah lainnya. Menurut Organisasi Kesehatan Dunia, 17,9 juta orang meninggal setiap tahunnya. Serangan jantung dan stroke menyebabkan lebih dari empat dari setiap lima kematian akibat penyakit kardiovaskular, dan sepertiga dari kematian ini terjadi sebelum usia 70 tahun. Sebuah database komprehensif mengenai faktor-faktor yang berkontribusi terhadap serangan jantung. ")
        @st.cache_data
        def convert_df(dataframe):
            return df.to_csv().encode('utf-8')
        csv = convert_df(df)

        st.download_button(
            label="download File csv",
            data=csv,
            file_name="Dataset.csv",
            mime="text/csv",
        )
    st.header("Tujuan dari dataset ini")
    st.write("Tujuan utamanya di sini adalah untuk mengumpulkan karakteristik Serangan Jantung atau faktor-faktor yang berkontribusi terhadapnya. Dataset berukuran 1319 sampel yang mempunyai sembilan field, dimana delapan field untuk field input dan satu field untuk field output. Usia, jenis kelamin (0 untuk Wanita, 1 untuk Pria), detak jantung (impuls), TD sistolik (tekanan tinggi), TD diastolik (tekanan rendah), gula darah (glukosa), CK-MB (kcm), dan Test-Troponin (troponin ) mewakili field input, sedangkan field output berkaitan dengan adanya serangan jantung (kelas), yang dibagi menjadi dua kategori (negatif dan positif); negatif mengacu pada tidak adanya serangan jantung, sedangkan positif mengacu pada adanya serangan jantung.")
    st.write(" Link dataset : https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset")
    st.header("Evaluasi dari Model ")
    col15,col16 = st.columns(2)
    with col15:
        st.image("distribusidata.jpg")
