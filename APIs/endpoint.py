import flask
from flask import request, jsonify
from model.predict import Predict

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/api/v1/<country>/predict', methods=['POST'])
def predict_for_country(country):
  parameters = request.json
  predict_pipeline = Predict(country)
  predictions = predict_pipeline.predict_for_a_period(parameters['start_date'], parameters['end_date'])
  response = jsonify(predictions)
  response.headers.add('Access-Control-Allow-Origin', '*')
  return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
