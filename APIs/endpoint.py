import flask
from flask import request, jsonify
from flask_cors import CORS

from model.country_data import CountryData
from model.predict import Predict

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api/v1/<country>/predict', methods=['POST'])
def predict_for_country(country):
  parameters = request.json
  predict_pipeline = Predict(country, window_size=7)
  predictions = predict_pipeline.predict_for_a_period(parameters['start_date'], parameters['end_date'])
  response = jsonify(predictions)
  return response


@app.route('/api/v1/<country>/get_constant_data', methods=['GET'])
def get_constant_data(country):
  cd = CountryData(country)
  return jsonify(cd.get_constant_features())


@app.route('/api/v1/<country>/get_variable_data', methods=['GET'])
def get_variable_data(country):
  start_date = request.args.get('start_date')
  end_date = request.args.get('end_date')
  print(start_date, end_date)
  cd = CountryData(country)
  policies_df = cd.get_policies_for_a_period(start_date, end_date)
  # convert the index, which is a date, into a string
  dates = policies_df.index.astype(str).values.tolist()
  format_to_return = {"dates": dates, "policies": {}}
  for policy in policies_df.columns:
    format_to_return["policies"][policy] = policies_df[policy].values.tolist()

  return jsonify(format_to_return)


# app route predict for country personalized features
@app.route('/api/v1/<country>/predict_personalized', methods=['POST'])
def predict_for_country_personalized(country):
  parameters = request.json
  predict_pipeline = Predict(country, window_size=7)
  predictions = predict_pipeline.predict_for_a_period_personalized(parameters['start_date'], parameters['end_date'],
                                                                   parameters['features'])
  response = jsonify(predictions)
  return response

# app route get shap values for a country
@app.route('/api/v1/<country>/get_shap_values', methods=['GET'])
def get_shap_values(country):
  cd = CountryData(country)
  return jsonify(cd.get_shap_for_country())

# app route get countries
@app.route('/api/v1/get_countries', methods=['GET'])
def get_countries():
  cd = CountryData('CHE') # CHE is a dummy country to get all countries
  return jsonify(cd.get_all_countries())

#app route predict_for_country_economic
@app.route('/api/v1/<country>/predict_economic', methods=['GET'])
def predict_for_country_economic(country):
  predict_pipeline = Predict(country, window_size=28, economic=True)
  predictions = predict_pipeline.predict_for_a_period("2020-04-01", "2021-05-31")
  response = jsonify(predictions)
  return response

# app route predict for country economic personalized
@app.route('/api/v1/<country>/predict_economic_personalized', methods=['POST'])
def predict_for_country_economic_personalized(country):
  parameters = request.json
  predict_pipeline = Predict(country, window_size=28, economic=True)
  predictions = predict_pipeline.predict_for_a_period_personalized(parameters['start_date'], parameters['end_date'],
                                                                   parameters['features'])
  response = jsonify(predictions)
  return response
