import numpy as np
import predict1
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('finalized_xgb_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

   # int_features = [int(x) for x in request.form.values()]
    ticker = request.form['ticker']
    result= predict1.make_prediction(ticker)
    print(result)
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    return render_template('index.html', prediction_text='The predicted prizes for {} is : Close ${} and Open ${}.'.format(ticker, "%.2f" % result[0], "%.2f" % result[1]))

# @app.route('/results',methods=['POST'])
# def results():
#
#     data = request.get_json(force=True)
#     # prediction = model.predict([np.array(list(data.values()))])
#     #
#     # output = prediction[0]
#     ticker = data['name']
#     result= predict1.make_prediction(ticker)
#     return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)