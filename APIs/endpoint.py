import flask
from flask import request, jsonify
import pandas as pd
from model.country_data import CountryData
from model.predict import Predict
from flask_cors import CORS

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api/v1/<country>/predict', methods=['POST'])
def predict_for_country(country):
    parameters = request.json
    predict_pipeline = Predict(country)
    predictions = predict_pipeline.predict_for_a_period(parameters['start_date'], parameters['end_date'])
    response = jsonify(predictions)
    return response

@app.route('/api/v1/<country>/get_constant_data', methods=['GET'])
def get_constant_data(country):
    cd = CountryData()
    return jsonify(cd.get_constant_features(country))

@app.route('/api/v1/<country>/get_variable_data', methods=['GET'])
def get_variable_data(country):
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    print(start_date, end_date)
    cd = CountryData()
    policies_df = cd.get_policies_for_a_period(country, start_date, end_date)
    #convert the index, which is a date, into a string
    dates = policies_df.index.astype(str).values.tolist()
    format_to_return = {"dates": dates, "policies": {}}
    for policy in policies_df.columns:
        format_to_return["policies"][policy] = policies_df[policy].values.tolist()

    return jsonify(format_to_return)

# app route predict for country personalized features
@app.route('/api/v1/<country>/predict_personalized', methods=['POST'])
def predict_for_country_personalized(country):
    parameters = request.json
    predict_pipeline = Predict(country)
    predictions = predict_pipeline.predict_for_a_period_personalized(parameters['start_date'], parameters['end_date'],
                                                                     parameters['features'])
    response = jsonify(predictions)
    return response


