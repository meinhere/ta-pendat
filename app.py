from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
# load model
model_full = pickle.load(open('model/model_full_pkl', 'rb'))
model_neuro = pickle.load(open('model/model_neuro_pkl', 'rb'))
model_fisio = pickle.load(open('model/model_fisio_pkl', 'rb'))

fisio_encode = {
    "hair_phenotype": {
        "Curly_hair": 0,
        "Wavy_hair": 3,
        "Straight_hair": 2,
        "No_hair": 1
    },
    "heart_rate": {
        "Medium_PulseRate": 2,
        "High_PulseRate": 0,
        "Low_PulseRate": 1
    },
    "skin_conductance": {
        "Normal_Conductance": 2,
        "Low_Conductance": 1,
        "High_Conductance": 0
    },
    "skin_temperature": {
        "Normal_Temperature": 2,
        "Fever": 0,
        "Low_Temperature": 1
    },
    "cortisol_level": {
        "AverageCL": 1,
        "Below_AverageCL": 2,
        "Above_AverageCL": 0
    },
    "systolic_bp": {
        "Range2_LowSystolic": 1,
        "Range3_LowSystolic": 2,
        "Range1_LowSystolic": 0
    },
    "diastolic_bp": {
        "NormalDiSystolic": 1,
        "LowDiSystolic": 0,
        "VerylowDiSystolic": 2
    }
}

target_encode = {
    2: "Sedang",
    1: "Rendah",
    0: "Tinggi",
}

style_encode = {
    2: ["bg-yellow-600 text-yellow-50", "bg-yellow-100 text-yellow-700 hover:bg-yellow-200"],
    1: ["bg-red-600 text-red-50", "bg-red-100 text-red-700 hover:bg-red-200"],
    0: ["bg-green-600 text-green-50", "bg-green-100 text-green-700 hover:bg-green-200"],
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    signal_amplitudo = int(request.form['signal_amplitudo'])
    delta_band = float(request.form['delta_band'])
    theta_band = float(request.form['theta_band'])
    alpha_band = float(request.form['alpha_band'])
    beta_band = float(request.form['beta_band'])
    hair_phenotype = fisio_encode['hair_phenotype'][request.form['hair_phenotype']]
    heart_rate = fisio_encode['heart_rate'][request.form['heart_rate']]
    skin_conductance = fisio_encode['skin_conductance'][request.form['skin_conductance']]
    skin_temperature = fisio_encode['skin_temperature'][request.form['skin_temperature']]
    cortisol_level = fisio_encode['cortisol_level'][request.form['cortisol_level']]
    systolic_bp = fisio_encode['systolic_bp'][request.form['systolic_bp']]
    diastolic_bp = fisio_encode['diastolic_bp'][request.form['diastolic_bp']]

    list_data = [signal_amplitudo, delta_band, theta_band, alpha_band, beta_band, hair_phenotype, heart_rate, skin_conductance, skin_temperature, cortisol_level, systolic_bp, diastolic_bp]

    prediction = model_full.predict([list_data])
    label_class = target_encode[prediction[0]]
    style_class = style_encode[prediction[0]]
    return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)