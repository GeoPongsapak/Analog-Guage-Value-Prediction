from flask import Flask, jsonify, request
from utils import acw, full_circle
from config.configuration import *
from os.path import join
import pickle



app = Flask(__name__)

map_gauge = dict({
    "1":"half_circle",
    "2":"full-circle"
})

@app.route("/")
def hello():
    return "hello"

@app.route("/predict", methods=["POST"])
def predict():
    imagefile = request.files['image']
    min_value = request.form.get('min')
    max_value = request.form.get('max')
    gauge_id = request.form.get('gauge_id')
    
    npimg = np.frombuffer(imagefile.read(), np.uint8)
    # print
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

    gauge_type = map_gauge[gauge_id]

    res = {}
    if gauge_type == "full-circle":
        print("full cycle type")
        print(img.shape)
        predict = full_circle.ValuePredict(
            frame=img,
            end_value=float(max_value),
            start_value=float(min_value)
        )
        coor_needle = predict.show_result(draw=True)
        print(coor_needle)
        res.update({"needle":coor_needle})
        val = round(predict.predicted_value,2)
        res.update({"value":val})

        return res

    return "OK"
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)