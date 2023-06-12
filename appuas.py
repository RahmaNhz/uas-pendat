import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


# import warnings
# warnings.filterwarnings("ignore")


st.title("DATA MINING")
st.write("##### Nama  : Rahma Nurhaliza")
st.write("##### Nim   : 210411100176")
st.write("##### Kelas : Penambangan Data B ")

description, upload_data, preprocessing, modeling, implementation = st.tabs(["Description", "Upload Data", "Preprocessing", "Modeling", "Implementation"])

with description:
    st.write("###### Aplikasi ini digunakan untuk memprediksi seseorang beresiko kecil atau besar untuk terkena serangan jantung dengan menggunakan beberapa metode seperti Naive Bayes,K-NN, Decision Tree & MLP ")
    st.write("""# Dekripsi Dataset """)
    st.write("###### Data Set Ini Adalah : Heart Attack Analysis & Predicition Dataset (Prediksi Serangan Jantung) ")
    st.write("###### Sumber Dataset ini diambil dari Kaggle :https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset")
    st.write(" Dataset ini dapat digunakan untuk memprediksi kemungkinan kecil seseorang terserang penyakit jantung atau lebih banyak kemungkinan terserang penyakit jantung . ")
    st.write("Terdapat Sebanyak 302 data tipe numerik yang digunakan untuk memprediksi kemungkinan besar atau kecil seseorang terserang penyakit penyakit jantung. ")
    st.write("""# Deskripsi Data""")
    st.write("Total datanya adalah 302 data, dan terdapat 13 atribut")
    st.write("Informasi Atribut")
    st.write("1) age: Usia pasien ")
    st.write("2) sex : Jenis kelamin pasien, 0= laki-laki & 1= perempuan ")
    st.write("3) trtbps : tekanan darah istirahat (dalam mm Hg)  ")
    st.write("4) chol : cholestoral dalam mg/dl diambil melalui sensor BMI ")
    st.write("5) fbs : (gula darah puasa > 120 mg/dl) (1 = benar; 0 = salah) ")
    st.write("6) restecg : hasil elektrokardiografi istirahat. nilai 0= normal, nilai 1= kelainan gelombang ST-T, nilai 2: menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri ")
    st.write("7) thalachh : detak jantung maksimum yang dicapai ")
    st.write("8) exng :angina akibat olahraga (1 = ya; 0 = tidak) ")
    st.write("9) oldpeak :  pengukuran perubahan depresi segmen ST pada elektrokardiogram (EKG) setelah aktivitas fisik dibandingkan dengan tingkat istirahat.(mm)")
    st.write("10) slp : slope (kecuraman) angka 0= tidak adanya perubahan segemen ST yang signifikan pada EKG pasien, angka 1= menunjukkan elevasi segemen ST pada EKG pasien, angka 2 = Mengindikasi depresi segmen ST pada EKG pasien ")
    st.write("11) caa : number of major vessels (0-3) ")
    st.write("12) thall : thalassemia, thall 1 = normal, thall 2 = tidak normal, thall 3 = abnormal/parah  ")
    st.write("13) cp : jenis nyeri dada dibagi menjadi berikut: ")
    st.write ("nilai 1 = angina tipikal")
    st.write ("nilai 2 = angina atipikal")
    st.write ("nilai 3 = nyeri non angina")
    st.write("nilai 4: tanpa gejala")
    st.write("##### Output:  ")
    st.write ("""0 = kecil/ sedikit kemungkinan serangan jantung""")
    st.write ("""1 = besar/lebih banyak kemungkinan serangan jantung""")

with upload_data:
    st.write("""# Dataset Asli """)
    df = pd.read_csv('https://raw.githubusercontent.com/RahmaNhz/Pendata/main/heart.csv')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
   
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['output'])
    y = df['output'].values
    df
    X
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('output Label')
    dumies = pd.get_dummies(df.output).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]]
    })

    st.write(labels)

with modeling:
    training, test = train_test_split(scaled_features,test_size=0.20, random_state=42)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.20, random_state=42)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        neural = st.checkbox('MLP')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=7
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        #ANNNBP
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=1)
        mlp.fit(training, training_label)
        mlp_pred = mlp.predict(test)
        mlp_akurasi = round(100 * accuracy_score(test_label, mlp_pred))


        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
            if neural:
                st.write("Model ANNBP accuracy score : {0:0.2f}".format(mlp_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi, mlp_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree','MLP'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
  ##Implementation
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        age = st.number_input('Masukkan umur : ')
        sex = st.number_input('Masukkan jenis kelamin 0=laki-laki & 1=perempuan : ')
        cp = st.number_input('Masukkan jenis nyeri dada 1-4  : ')
        trtbps = st.number_input('Masukkan tekanan darah istirahat  : ')
        chol = st.number_input('Masukkan Nilai cholestrol  : ')
        fbs = st.number_input('Masukkan nilai gula darah puasa 1=benar, 0=salah : ')
        restecg = st.number_input('Masukkan nilai restecg (0-2): ')
        thalach = st.number_input('Masukkan nilai thalach (1-3): ')
        exng= st.number_input('Masukkan nilai exng 1= iya, 0= tidak: ')
        oldpeak = st.number_input('Masukkan nilai oldpeak : ')
        slp = st.number_input('Masukkan nilai slp 0-2 : ')
        caa = st.number_input('Masukkan nilai caa 0-3: ')
        thall = st.number_input('Masukkan nilai thall 1-3: ')
        
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree','MLP'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                age,
                sex,	
                cp,	
                trtbps,	
                chol,	
                fbs,
                restecg,
                thalach,
                exng,
                oldpeak,
                slp,
                caa,
                thall
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt
            if model == 'MLP':
                mod = mlp

            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
