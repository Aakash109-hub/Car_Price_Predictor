from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('Linearreg.pkl', 'rb'))

car = pd.read_csv("Cleaned_car.csv")

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    company_model_map = {}
    for company in companies:
        company_model_map[company] = car[car['company'] == company]['name'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year,
                           fuel_types=fuel_type, company_model_map=company_model_map)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))  # ensure it's an integer
        fuel_type = request.form.get('fuel_type')
        driven = int(request.form.get('kilo_driven'))  # ensure it's an integer

        # Form the DataFrame to match your model's expected input
        data = pd.DataFrame([[car_model, company, year, driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        # Debug print to check what data looks like before prediction
        print(data)

        # Prediction
        prediction = model.predict(data)
        print(prediction)
        return str(np.round(prediction[0], 2))

    except Exception as e:
        # Return a detailed error message
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)