from flask import Flask,jsonify,request
from classifier import getPrediction
app=Flask(__name__)
@app.route("/predict",methods=["POST"])
def predict():
    image=request.files.get("image (1)")
    pred=getPrediction(image)
    return jsonify({
        "pred":pred
    })
if __name__=="__main__":
    app.run()