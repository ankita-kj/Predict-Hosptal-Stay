from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
from waitress import serve

app = Flask(__name__)

model = pickle.load(open('best_model_top_10_features.pkl', 'rb'))
imputer = SimpleImputer(strategy="median")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(request.form)

        input_features = [
            float(request.form['sykdom_underkategori_ARF/MOSF']),
            float(request.form['overlevelsesestimat_6mnd']),
            float(request.form['alder']),
            float(request.form['fysiologisk_score']),
            float(request.form['apache_fysiologisk_score']),
            float(request.form['overlevelsesestimat_2mnd']),
            float(request.form['dnr_status']),  
            float(request.form['hvite_blodlegemer']),
            float(request.form['lungefunksjon']),
            float(request.form['lege_overlevelsesestimat_6mnd'])
        ]

        input_df = pd.DataFrame([input_features])

        prediction = model.predict(input_df)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Predicted Length of Stay: {output} days')

    except ValueError as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction_text="Invalid input. Please enter numeric values for all fields.")

if __name__ == '__main__':
     serve(app, host='0.0.0.0', port=8080) 




