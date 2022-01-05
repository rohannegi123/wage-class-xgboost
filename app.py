from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle


app = Flask(__name__)

@app.route('/' , methods =['GET'])
@cross_origin()
def homepage():
    return render_template('index.html')

@app.route('/predict' , methods =['GET','POST'])
@cross_origin()
def Prediction():
    if request.method == 'POST':
        try:
            age = request.form['age']
            workclass = request.form['workclass']
            fnlwgt = request.form['fnlwgt']
            education = request.form['education']
            education_num = request.form['education_num']
            marital_status = request.form['marital_status']
            occupation = request.form['occupation']
            relationship = request.form['relationship']
            race = request.form['race']
            sex = request.form['sex']
            capital_gain = request.form['capital_gain']
            capital_loss = request.form['capital_loss']
            hours_per_week = request.form['hours_per_week']
            native_country = request.form['native_country']
            filename = 'xgboost-assign.pickle'
            load_model = pickle.load(open(filename, 'rb'))
            standfile = 'sd_sc_xgboost'
            Stand = pickle.load(open(standfile,'rb'))
            ans = load_model.predict(Stand.transform([[age,workclass, fnlwgt,education,education_num,marital_status, occupation,relationship,race, sex, capital_gain,capital_loss,hours_per_week,native_country]]))
            if ans == 1:
                prediction = 'The wage class of the person is greater than 50K'

            else:
                prediction = 'The wage class of the person is <= 50K'
            return render_template('results.html', prediction=prediction )
        except Exception as e:
            print('the exception msg is : ' , e)
            return 'Something went wrong'
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug =True)