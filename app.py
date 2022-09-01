from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('mrfcoracao.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trtbps = int(request.form['trtbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])
    values = np.array([[age,sex,cp,trtbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])

    prediction = model.predict(values)
    
    if(prediction==1):
        return render_template('positiveresult.html')
    elif(prediction==0):
        return render_template('negativeresult.html')
 
if __name__=="__main__":
    app.run(host='0.0.0.0', port='5000')