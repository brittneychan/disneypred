from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # allows the web interface to call it

rf = pickle.load(open('rf_model.pkl', 'rb'))
xgb = pickle.load(open('xgb_model.pkl', 'rb'))
model = pickle.load(open('lr_model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ride = data['ride']
        date_str = data['date']
        hour = int(data['hour'])
        mean_temp = float(data['mean_temp'])
        precipitation = float(data['precipitation'])
        season = data['season']

        date = pd.to_datetime(date_str)

        input_data = {
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'hour': hour,
            'day_of_week': date.dayofweek,
            'precipitation': precipitation,
            'mean temp': mean_temp,
            'max temp': mean_temp + 5,
            'min temp': mean_temp - 5,
            'is_weekend': int(date.dayofweek in [5, 6]),
            'temp_range': 10,
            'month_sin': np.sin(2 * np.pi * date.month / 12),
            'month_cos': np.cos(2 * np.pi * date.month / 12),
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'week_of_year': date.isocalendar()[1],
            'day_of_year': date.dayofyear,
            'ride': ride,
            'season': season
        }

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df, columns=['ride', 'season'])
        input_df = input_df.reindex(columns=columns, fill_value=0)

        rf_pred = round(float(rf.predict(input_df)[0]), 1)
        xgb_pred = round(float(xgb.predict(input_df)[0]), 1)
        lr_pred = round(float(model.predict(input_df)[0]), 1)
        avg = round((rf_pred + xgb_pred) / 2, 1)

        return jsonify({
            'rf': rf_pred,
            'xgb': xgb_pred,
            'lr': lr_pred,
            'avg': avg
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)