from flask import Flask, jsonify, request, render_template, url_for, flash, redirect
from predictionModel import PredictionModel
import pandas as pd
from random import randrange

app = Flask(__name__,
            template_folder="./template")
app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506'


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        content = request.form['content']
        if not content:
            flash('NEWS Content is required!')
        else:
            model = PredictionModel(content)
            prediction = model.predict()
            # return redirect(url_for('home', form_content=content))
            return render_template('index.html', form_content=content, prediction=prediction)
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    model = PredictionModel(request.json['text'])
    return jsonify(model.predict())


# Only for local running
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
