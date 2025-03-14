from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)

model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car = pd.read_csv("Cleaned_quikr_car.csv")

# @app.route('/')

# def print_hi(name):
#     print(f'Hi, {name}')
# if __name__ == '__main__':
#     print_hi('PyCharm')


@app.route('/' ,methods=['GET','POST'])
def index():
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    year = sorted(car["year"].unique(), reverse=True)
    fuel_type = car["fuel_type"].unique()
    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))
    print(company, car_model, year, fuel_type, kms_driven)

    prediction = model.predict(pd.DataFrame([[car_model, company, year, fuel_type, kms_driven]], columns=['name', 'company', 'year', 'fuel_type', 'z']))
    # print(prediction)
    # prediction = model.predict(pd.DataFrame(columns=['company', 'name', 'year', 'fuel_type', 'kms_driven'],
    #                                         data = np.array([[company, car_model, year, fuel_type, kms_driven]].reshape(1, 5))
    # ))
    print("=========================" + str(prediction) + "=========================" )
    # prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
    #                                         data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))

    return str(np.round(prediction[0],2))

if __name__ == '__main__':
    app.run(debug=True)