import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado y las columnas
modelo = joblib.load('modelo_quick_eat.pkl')
columnas_modelo = joblib.load('columnas_modelo.pkl')

st.title("🍔 QuickEat Delivery Pro: Predictor de Demoras")
st.write("Herramienta para estimar el riesgo de retrasos en Manizales.")

# Controles sencillos para ingresar datos
distancia = st.slider("Distancia estimada (km)", 0.5, 15.0, 3.2)
monto = st.number_input("Monto del pedido (COP)", value=35000)

if st.button("Evaluar Pedido"):
    datos_entrada = pd.DataFrame(0, index=[0], columns=columnas_modelo)
    if 'distancia_km' in columnas_modelo:
        datos_entrada['distancia_km'] = distancia
    if 'monto_pedido_cop' in columnas_modelo:
        datos_entrada['monto_pedido_cop'] = monto

    prediccion = modelo.predict(datos_entrada)[0]
    probabilidad = modelo.predict_proba(datos_entrada)[0][1]

    if prediccion == 1:
        st.error(f"⚠️ Alerta: Alto riesgo de demora. Probabilidad: {probabilidad:.2%}")
    else:
        st.success(f"✅ Normal: Pedido a tiempo. Probabilidad de demora: {probabilidad:.2%}")
