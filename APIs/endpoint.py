import flask
from flask import request, jsonify
import pandas as pd
from model.country_data import CountryData
from model.predict import Predict
from flask_cors import CORS

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app, resources={r"/api/*": {"origins": "http://localhost:4200"}})


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
    parameters = request.json
    cd = CountryData()
    policies_df = cd.get_policies_for_a_period(country, parameters['start_date'], parameters['end_date'])
    #convert the index, which is a date, into a string
    policies_df.index = policies_df.index.astype(str)
    format_to_return = {"dates": policies_df.index.values.tolist(), "policies": {}}
    for policy in policies_df.columns:
        format_to_return["policies"][policy] = policies_df[policy].values.tolist()

    return jsonify(format_to_return)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
