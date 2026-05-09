# app.py
import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from utils import load_and_preprocess_data, evaluate_regression
from sklearn.preprocessing import StandardScaler
import json

app = Flask(__name__)
from datetime import datetime

@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}

# Load models and scalers
models = {}
scaler = None
scaler_seq = None
feature_cols = ['TMAX_lag1','TMAX_lag2','TMAX_lag3','TMIN_lag1','TMIN_lag2','TMIN_lag3',
                'PRCP_lag1','PRCP_lag2','PRCP_lag3','RAIN_lag1','RAIN_lag2','RAIN_lag3',
                'TMIN','PRCP','RAIN']

# Load dataset untuk query historis
df_original = None

try:
    scaler = joblib.load('models/scaler.pkl')
    scaler_seq = joblib.load('models/scaler_seq.pkl')
    models['lr'] = joblib.load('models/linear_regression.pkl')
    models['ann'] = tf.keras.models.load_model('models/ann_model.h5')
    models['lstm'] = tf.keras.models.load_model('models/lstm_model.h5')
    models['bp'] = tf.keras.models.load_model('models/backprop_model.h5')
    models['kmeans'] = joblib.load('models/kmeans.pkl')
    with open('models/results.json', 'r') as f:
        results = json.load(f)

    # Load original dataset untuk query historis
    df_original = pd.read_csv('seattleWeather_1948-2017.csv')
    df_original['DATE'] = pd.to_datetime(df_original['DATE'])
    df_original = df_original.sort_values('DATE').reset_index(drop=True)
    df_original['RAIN'] = df_original['RAIN'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    # Isi missing value jika ada (beberapa tanggal mungkin NA)
    df_original['RAIN'] = df_original['RAIN'].fillna(0)
    # Drop baris dengan NaN di TMAX/TMIN/PRCP jika ada
    df_original = df_original.dropna(subset=['TMAX', 'TMIN', 'PRCP'])
    print("Dataset historis berhasil dimuat.")
except Exception as e:
    print(f"Error loading models/dataset: {e}")
    models = {}
    results = {}

def get_features_for_date(target_date, model_choice='lr'):
    """
    Mengambil fitur untuk tanggal tertentu berdasarkan 3 hari sebelumnya dari dataset.
    Mengembalikan prediksi TMAX untuk target_date.
    """
    # Cari indeks target_date
    target_row = df_original[df_original['DATE'] == pd.to_datetime(target_date)]
    if target_row.empty:
        return None, "Tanggal tidak ditemukan dalam dataset."
    
    target_idx = target_row.index[0]
    if target_idx < 3:
        return None, "Data 3 hari sebelum tanggal ini tidak tersedia."

    # Ambil data 3 hari sebelumnya
    rows = df_original.iloc[target_idx-3:target_idx]  # 3 hari: lag3, lag2, lag1 (dari yang terlama)
    if len(rows) < 3:
        return None, "Data tidak lengkap."

    # Pastikan urutan: hari H-3, H-2, H-1
    row_lag3 = rows.iloc[0]  # paling lama
    row_lag2 = rows.iloc[1]
    row_lag1 = rows.iloc[2]  # paling baru (kemarin)

    # Ambil fitur untuk prediksi
    tmax_lag1 = row_lag1['TMAX']
    tmax_lag2 = row_lag2['TMAX']
    tmax_lag3 = row_lag3['TMAX']
    tmin_lag1 = row_lag1['TMIN']
    tmin_lag2 = row_lag2['TMIN']
    tmin_lag3 = row_lag3['TMIN']
    prcp_lag1 = row_lag1['PRCP']
    prcp_lag2 = row_lag2['PRCP']
    prcp_lag3 = row_lag3['PRCP']
    rain_lag1 = row_lag1['RAIN']
    rain_lag2 = row_lag2['RAIN']
    rain_lag3 = row_lag3['RAIN']

    # Fitur hari ini (sebenarnya target_date adalah hari ini? Kita prediksi TMAX di target_date,
    # jadi fitur 'hari ini' adalah data pada target_date (seolah-olah kita sudah tahu TMIN, PRCP, RAIN di hari yang sama)
    # Di dataset, untuk memprediksi TMAX hari ini, kita menggunakan TMIN, PRCP, RAIN hari ini.
    hari_ini = df_original.iloc[target_idx]
    tmin_today = hari_ini['TMIN']
    prcp_today = hari_ini['PRCP']
    rain_today = hari_ini['RAIN']

    # Buat feature vector
    features = np.array([[tmax_lag1, tmax_lag2, tmax_lag3,
                          tmin_lag1, tmin_lag2, tmin_lag3,
                          prcp_lag1, prcp_lag2, prcp_lag3,
                          rain_lag1, rain_lag2, rain_lag3,
                          tmin_today, prcp_today, rain_today]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Prediksi menggunakan model yang dipilih
    if model_choice == 'lr':
        pred = models['lr'].predict(features_scaled)[0]
    elif model_choice == 'ann':
        pred = models['ann'].predict(features_scaled)[0][0]
    elif model_choice == 'lstm':
        # Untuk LSTM, perlu sequence 3 hari
        seq_features = np.array([[tmax_lag3, tmin_lag3, prcp_lag3, rain_lag3],
                                 [tmax_lag2, tmin_lag2, prcp_lag2, rain_lag2],
                                 [tmax_lag1, tmin_lag1, prcp_lag1, rain_lag1]])
        seq_scaled = scaler_seq.transform(seq_features)
        seq_input = seq_scaled.reshape(1, 3, 4)
        pred = models['lstm'].predict(seq_input)[0][0]
    elif model_choice == 'bp':
        pred = models['bp'].predict(features_scaled)[0][0]
    else:
        return None, "Model tidak valid."

    return pred, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    error = None
    selected_date = None
    selected_model = 'lr'
    actual_tmax = None

    if request.method == 'POST':
        selected_date = request.form.get('date')
        selected_model = request.form.get('model', 'lr')

        # Validasi tanggal
        if not selected_date:
            error = "Silakan pilih tanggal."
        else:
            pred, err = get_features_for_date(selected_date, selected_model)
            if err:
                error = err
            else:
                prediction = round(pred, 1)
                # Ambil TMAX aktual jika ada di dataset
                target_row = df_original[df_original['DATE'] == pd.to_datetime(selected_date)]
                if not target_row.empty:
                    actual_tmax = target_row.iloc[0]['TMAX']

    return render_template('predict.html',
                           prediction=prediction,
                           error=error,
                           selected_date=selected_date,
                           selected_model=selected_model,
                           actual_tmax=actual_tmax)

@app.route('/comparison')
def comparison():
    reg_results = {k: v for k, v in results.items() if 'MAE' in v}
    return render_template('comparison.html', results=reg_results)

@app.route('/api/results')
def api_results():
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)