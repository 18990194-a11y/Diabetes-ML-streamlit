import streamlit as st
import numpy as np
import joblib

modelo = joblib.load("modelo_logistic.pkl")

# título
st.title("Predicción de Diabetes")

# datos del estudiante
st.write("**Nombre:** Rosa Maria Flores Echeverria")
st.write("**Código ISIL:** 18990194")

# link colab
st.markdown("[Ver Cuaderno de Google Colab](https://colab.research.google.com/drive/1JdT-Hm3E-ZHROKLC3wKCUxr3-h3Gs2f7?usp=sharing)")

st.write("---")

st.subheader("Ingrese los datos del paciente")

# inputs
pregnancies = st.number_input("Pregnancies", min_value=0.0)
glucose = st.number_input("Glucose", min_value=0.0)
bloodpressure = st.number_input("BloodPressure", min_value=0.0)
skinthickness = st.number_input("SkinThickness", min_value=0.0)
insulin = st.number_input("Insulin", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0)
age = st.number_input("Age", min_value=0.0)

# botón de predicción
if st.button(" Predecir"):
    datos = np.array([[pregnancies,
                       glucose,
                       bloodpressure,
                       skinthickness,
                       insulin,
                       bmi,
                       dpf,
                       age]])

    pred = modelo.predict(datos)

    st.write("---")
    st.subheader(" Resultado")

    if pred[0] == 1:
        st.error("El paciente podría tener diabetes")
    else:
        st.success("El paciente no tendría diabetes")