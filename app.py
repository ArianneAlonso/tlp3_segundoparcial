import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Cargar dataset
data_set = pd.read_csv("anemia.csv")
data_set.replace(['NaN', ' '], np.nan, inplace=True)
data_set.dropna(inplace=True)

data_set['Result'] = data_set['Result'].map({1: 1, 0: 0})

X = data_set.drop(columns='Result')
y = data_set['Result']

X_mean = X.mean()
X_std = X.std()
X_scaled = (X - X_mean) / X_std

# Modelo de Keras
modelo = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])
modelo.compile(loss="binary_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"])
history = modelo.fit(X_scaled, y, epochs=100, batch_size=16, validation_split=0.2, verbose=0)

def plot_loss():
    plt.figure()
    plt.plot(history.history['loss'], label='Pérdida entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida validación')
    plt.title('Gráfico de pérdida')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_regression(gender=0):
    hemoglobin_range = np.linspace(X['Hemoglobin'].min(), X['Hemoglobin'].max(), 100)
    mch_avg = X['MCH'].mean()
    mchc_avg = X['MCHC'].mean()
    mcv_avg = X['MCV'].mean()
    
    inputs = np.array([[gender, h, mch_avg, mchc_avg, mcv_avg] for h in hemoglobin_range])
    inputs_scaled = (inputs - X_mean.values) / X_std.values
    preds = modelo.predict(inputs_scaled).flatten()
    
    plt.figure()
    plt.plot(hemoglobin_range, preds, label='Probabilidad predicha')
    plt.scatter(X['Hemoglobin'], y, alpha=0.3, label='Datos reales')
    plt.title(f'Regresión logística: Hemoglobina vs Probabilidad (Género={gender})')
    plt.xlabel('Hemoglobina (g/dL)')
    plt.ylabel('Probabilidad de anemia')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    loss_img = plot_loss()
    reg_img = plot_regression()
    return render_template('index.html', prediction=False, loss_img=loss_img, reg_img=reg_img, section='info')

@app.route('/predict', methods=['POST'])
def predict():
    gender = 0 if request.form['gender'] == "Femenino" else 1
    hemoglobin = float(request.form['hb'])
    mch = float(request.form['mch'])
    mchc = float(request.form['mchc'])
    mcv = float(request.form['mcv'])

    input_data = np.array([gender, hemoglobin, mch, mchc, mcv])
    input_scaled = (input_data - X_mean.values) / X_std.values
    prob = modelo.predict(input_scaled.reshape(1, -1))[0][0]
    result = "Anémico" if prob >= 0.5 else "Normal"

    loss_img = plot_loss()
    reg_img = plot_regression(gender)
    
    return render_template('index.html', 
                           prediction=True, 
                           probabilidad=f"{prob:.2%}", 
                           clasificacion=result,
                           loss_img=loss_img,
                           reg_img=reg_img,
                           section='pred')

if __name__ == '__main__':
    app.run(debug=True)