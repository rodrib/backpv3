from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi import FastAPI
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import numpy as np


app = FastAPI()

# Configurar CORS para permitir todas las solicitudes durante el desarrollo local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Cargar el DataFrame desde el archivo CSV existente
df = pd.read_csv("df_completo1.csv")  # Ajusta el nombre del archivo según sea necesario

# Ruta para la raíz
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# Ruta para obtener datos del DataFrame
@app.get("/api/data")
def get_data():
    return df.to_dict(orient="records")

###
#### Predicciones
from fastapi import FastAPI, Query
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



# Crear DataFrame de ejemplo
data = {
    'RECIDIVA': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'DX1': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'EDAD': [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 55, 60, 65, 70, 75],
    'GRADO1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    'HER21': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df_ejemplo = pd.DataFrame(data)

# Definir las características (X) y las etiquetas (y)
X = df_ejemplo[['EDAD', 'GRADO1']]
y_recidiva = df_ejemplo['RECIDIVA']
y_dx1 = df_ejemplo['DX1']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_recidiva, X_test_recidiva, y_train_recidiva, y_test_recidiva = train_test_split(X, y_recidiva, test_size=0.2, random_state=42)
X_train_dx1, X_test_dx1, y_train_dx1, y_test_dx1 = train_test_split(X, y_dx1, test_size=0.2, random_state=42)

# Ruta de la API para el modelo de recurrencia del cáncer
@app.get("/api/recidiva")
def predict_recidiva():
    # Modelo de Regresión Logística para predecir la recurrencia del cáncer
    model_recidiva = LogisticRegression()
    model_recidiva.fit(X_train_recidiva, y_train_recidiva)
    y_pred_recidiva = model_recidiva.predict(X_test_recidiva)
    report_recidiva = classification_report(y_test_recidiva, y_pred_recidiva, output_dict=True)
    accuracy_recidiva = model_recidiva.score(X_test_recidiva, y_test_recidiva)
    return {"report": report_recidiva, "accuracy": accuracy_recidiva}

# Ruta de la API para el modelo de diagnóstico del paciente
@app.get("/api/dx1")
def predict_dx1():
    # Modelo de Regresión Logística para predecir el diagnóstico del paciente
    model_dx1 = LogisticRegression()
    model_dx1.fit(X_train_dx1, y_train_dx1)
    y_pred_dx1 = model_dx1.predict(X_test_dx1)
    report_dx1 = classification_report(y_test_dx1, y_pred_dx1, output_dict=True)
    accuracy_dx1 = model_dx1.score(X_test_dx1, y_test_dx1)
    return {"report": report_dx1, "accuracy": accuracy_dx1}

# Ruta de la API para hacer predicciones con los modelos
from fastapi.encoders import jsonable_encoder

# Ruta de la API para hacer predicciones con los modelos
@app.get("/api/prediccion")
def hacer_prediccion(edad: int = Query(...), grado: int = Query(...)):
    # Aquí puedes utilizar los modelos entrenados para hacer las predicciones
    # Por ejemplo, podrías utilizar los modelos de recurrencia del cáncer y diagnóstico del paciente

    # Predicción de recurrencia del cáncer
    model_recidiva = LogisticRegression()
    model_recidiva.fit(X_train_recidiva, y_train_recidiva)
    recidiva_prediccion = model_recidiva.predict([[edad, grado]])
    
    # Predicción de diagnóstico del paciente
    model_dx1 = LogisticRegression()
    model_dx1.fit(X_train_dx1, y_train_dx1)
    dx1_prediccion = model_dx1.predict([[edad, grado]])

    # Formatear el resultado de la predicción
    resultado_prediccion = {
        "edad": int(edad),  # Convertir a int
        "grado": int(grado),  # Convertir a int
        "recidiva_prediccion": int(recidiva_prediccion[0]),  # Convertir a int
        "dx1_prediccion": int(dx1_prediccion[0])  # Convertir a int
    }

    # Convertir el resultado a JSON
    resultado_json = jsonable_encoder(resultado_prediccion)

    return resultado_json


####################
""" from sklearn.cluster import KMeans
import numpy as np

# Carga el modelo de clustering preentrenado
kmeans_model = KMeans(n_clusters=3, random_state=42)  # Por ejemplo, con 3 clusters

# Datos de ejemplo para el modelo de clustering
X = np.array([[0, 1, 45, 1, 0], [1, 0, 50, 2, 1], [0, 1, 55, 1, 0], [1, 0, 60, 2, 1], [0, 1, 65, 1, 0], [1, 0, 70, 2, 1],
              [0, 1, 75, 1, 0], [1, 0, 80, 2, 1], [0, 1, 85, 1, 0], [1, 0, 90, 2, 1], [0, 1, 55, 1, 0], [1, 0, 60, 2, 1],
              [0, 1, 65, 1, 0], [1, 0, 70, 2, 1], [0, 1, 75, 1, 0]])

# Entrena el modelo de clustering con los datos de ejemplo
kmeans_model.fit(X)

@app.get("/api/prediccion_cluster")
def predecir_cluster(RECIDIVA: int, DX1: int, EDAD: int, GRADO1: int, HER21: int):
    # Realiza la predicción del cluster con el modelo de clustering
    datos_usuario = np.array([[RECIDIVA, DX1, EDAD, GRADO1, HER21]])
    cluster_predicho = kmeans_model.predict(datos_usuario)[0]

    # Devuelve el cluster predicho al usuario
    return {"cluster_predicho": cluster_predicho} """

from fastapi import FastAPI, Query, HTTPException

import numpy as np
from fastapi import FastAPI, Query
import logging

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

@app.get("/api/correlacion")
def predecir_cluster(EDAD: int = Query(..., description="Edad del paciente"),
                     DX1: int = Query(..., description="Valor DX1 del paciente")):
    # Agregar registros para verificar los parámetros recibidos
    logging.debug("EDAD recibida: %s", EDAD)
    logging.debug("DX1 recibido: %s", DX1)
    
    try:
        # Calcula la correlación entre EDAD y DX1
        correlacion = np.corrcoef(np.array([[EDAD, DX1]]), rowvar=False)[0, 1]
    except Exception as e:
        # Si ocurre un error, registra el error
        logging.error("Error al calcular la correlación: %s", e)
        # Devuelve un mensaje de error al usuario
        raise HTTPException(status_code=500, detail="Error al calcular la correlación")

    # Agregar registros para verificar la correlación calculada
    logging.debug("Correlación calculada: %s", correlacion)

