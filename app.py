
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Cargar modelo y codificador
modelo = joblib.load('modelo_obesidad.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Imagen de cabecera
imagen = Image.open("fitness-header.jpg")
st.image(imagen, use_column_width=True)

# Título principal
st.title("Estimador de Nivel de Obesidad")
st.markdown("Esta aplicación predice el nivel estimado de obesidad a partir de tus datos personales y hábitos.")

# Secciones organizadas
st.markdown("## Datos personales")

col1, col2 = st.columns(2)

with col1:
    sexo = st.selectbox("Sexo", ["Male", "Female"])
    edad = st.slider("Edad", 14, 65, 25)
    altura = st.slider("Altura (m)", 1.4, 2.1, 1.7)
    peso = st.slider("Peso (kg)", 40, 180, 70)
    historial = st.selectbox("Antecedentes familiares de obesidad", ["yes", "no"])
    favc = st.selectbox("¿Consumes alimentos calóricos frecuentemente?", ["yes", "no"])

with col2:
    fcvc = st.slider("Frecuencia de consumo de verduras (0 a 3)", 0.0, 3.0, 2.0)
    ncp = st.slider("Número de comidas principales", 1.0, 4.0, 3.0)
    caec = st.selectbox("Comes entre comidas", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("¿Fumas?", ["yes", "no"])
    ch2o = st.slider("Litros de agua por día", 0.0, 3.0, 2.0)
    scc = st.selectbox("¿Controlas tu consumo calórico?", ["yes", "no"])
    faf = st.slider("Actividad física semanal (horas)", 0.0, 3.0, 1.0)
    tue = st.slider("Horas frente a pantalla por día", 0.0, 2.0, 1.0)
    calc = st.selectbox("Consumo de alcohol", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Medio de transporte", [
        "Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"
    ])

# Botón para predecir
if st.button("Predecir"):
    # Construir DataFrame con los datos ingresados
    entrada = pd.DataFrame([{
        'Gender': sexo,
        'Age': edad,
        'Height': altura,
        'Weight': peso,
        'fwo': historial,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans
    }])

    # Predicción
    pred = modelo.predict(entrada)
    clase = label_encoder.inverse_transform(pred)[0]

    st.subheader("Resultado")
    st.success(f"Nivel estimado de obesidad: **{clase}**")
    if sexo== 'Female':
        # Imagen según resultado (opcional)
        if "Obesity" in clase:
            st.image("obesidad.jpg", caption="Recomendamos seguimiento médico.", use_column_width=True)
        elif "Overweight" in clase:
            st.image("sobrepeso.jpg", caption="Hay señales de sobrepeso.", use_column_width=True)
        elif "Normal" in clase:
            st.image("fit.jpg", caption="Buen estado físico.", use_column_width=True)
        else:
            st.image("bajo_peso.jpg", caption="Posible bajo peso.", use_column_width=True)