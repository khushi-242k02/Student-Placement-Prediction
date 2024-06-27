from flask import Flask,render_template,request

import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_placement():
    cgpa = int(request.form.get('cgpa'))
    iq = int(request.form.get('iq'))
    profile_score = int(request.form.get('profile_score'))

    # prediction
    result = model.predict(np.array([cgpa,iq,profile_score]).reshape(1,3))
    # return str(result)

    if result[0] == 1:
        result = "Woohoo! Your chances of getting placed are higher."
    else:
        result = "Oops! Your chances of getting placed are not higher, you have to work hard."

    return render_template('index.html',result=result)
if __name__ == '__main__':
    app.run(debug=True)
