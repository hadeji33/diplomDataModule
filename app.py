import json

import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from flask import Flask, jsonify, request, abort

app = Flask(__name__)

filename = 'data/finalized_model.sav'
model = LogisticRegression()


def estimate(params):
    pickle.dump(model, open(filename, 'wb'))
    return model.predict_proba(params)


@app.route('/')
def index():
    testData = pd.read_csv('data/testdata.csv', ';')
    Xtest = testData[['X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
    return jsonify({"X": Xtest})


@app.route('/machine/train', methods=['GET'])
def train_machine():
    data = pd.read_csv('data/projects.csv', ';')
    X = data[['X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
    y = data['Y']

    model.fit(X, y)
    testData = pd.read_csv('data/testdata.csv', ';')
    Xtest = testData[['X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
    Ytest = testData['Y']
    parameters = model.coef_
    pickle.dump(model, open(filename, 'wb'))
    print(parameters)
    predicted_classes = model.predict(X)
    accuracy = accuracy_score(y.values.flatten(), predicted_classes)
    print(accuracy)
    y_test_pred = model.predict(Xtest)
    print(y_test_pred)
    y_ture = Ytest
    target_names = ['class 0', 'class 1']
    report = classification_report(y_ture, y_test_pred, target_names=target_names)
    print(report)
    return jsonify({"message": report})


@app.route('/machine/current-weights', methods=['GET'])
def current_weight():
    pickle.dump(model, open(filename, 'wb'))
    response = model.coef_
    return jsonify({"message": response})


@app.route('/projects/create', methods=['POST'])
def add_project():
    project_json = request.get_json()
    return jsonify({"message": "project added"}), 201


@app.route('/projects/estimate', methods=['POST'])
def estimate_project():
    if not request.json:
        abort(400)
    project_json = request.get_json()
    params = pd.DataFrame.from_dict(project_json, orient='index').T
    print(params)
    response = estimate(params)
    print(response[0])
    return jsonify({"result":  float(response[0][1])}), 201


if __name__ == '__main__':
    app.run()
