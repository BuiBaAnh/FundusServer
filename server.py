import flask
from flask_cors import CORS, cross_origin
from flask import request

from predict import predictODC
from service import experiment

app = flask.Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
@cross_origin()
def index():
    return "<h1>Welcome to our server !!</h1>" 


@app.route('/segOD', methods=['POST'])
@cross_origin()
def getDataOD():
    img = request.files["segOD"]
    ret = predictODC(img, 'OD')
    return ret

@app.route('/segOC', methods=['POST'])
@cross_origin()
def getDataOC():
    img = request.files["segOC"]
    ret = predictODC(img, 'OC')
    return ret


@app.route('/expOD', methods=['POST'])
@cross_origin()
def experimentDataOD():
    file = request.files["expOD"]
    ret = experiment(file, 'OD')
    return ret

@app.route('/expOC', methods=['POST'])
@cross_origin()
def experimentDataOC():
    file = request.files["expOC"]
    ret = experiment(file, 'OC')
    return ret

if __name__ == "__main__":
    app.run(port=5000)