from flask import Flask,render_template,request
import pickle 
import numpy as np
app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/salary")
def salary():
    return render_template("salary.html")

@app.route("/dashboard")
def dashboard():
    return render_template("tableaudashboard.html")

    
@app.route("/predict",methods=["POST"])
def predict():
    init_features=[float(x) for x in request.form.values()]
    final_features=[np.array(init_features)]

    prediction=model.predict(final_features)
    
    return render_template("result.html",prediction_text='Employee salary is {}'.format(prediction))


if __name__=='__main__':
    app.run()  
