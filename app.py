from flask import Flask,render_template,request
import pickle
import numpy as np
import sklearn
model = pickle.load(open('diabetes_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')

def diabetes():
    return render_template("diabetes.html")

@app.route('/Predict',methods=['POST'])

def predict_diabetes():
    Pregnancies = int(request.form.get('Pregnancies'))
    Glucose = int(request.form.get('Glucose'))
    BloodPressure = int(request.form.get('BloodPressure'))
    SkinThickness = int(request.form.get('SkinThickness'))
    Insulin = int(request.form.get('Insulin'))
    BMI = float(request.form.get('BMI'))
    DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
    Age = float(request.form.get('Age'))


    #Prediction

    result = model.predict(np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]).reshape(1,-1))
    if result[0]==0:
        result = 'Not Diabetic'
    else:
        result = 'Diabetic'
    return render_template('diabetes.html',result=result)
if __name__=='__main__':
    app.run(debug=True)