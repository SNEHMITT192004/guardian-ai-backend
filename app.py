# from re import DEBUG, sub
# from werkzeug.utils import secure_filename, send_from_directory
# import os
# import subprocess
# from flask import Flask, jsonify,request,redirect, send_file, url_for,render_template
# import joblib
# import numpy as np
# from sklearn.impute import SimpleImputer
# from flask_cors import CORS
# from sklearn.preprocessing import LabelEncoder
# import pandas as pd


# app = Flask(__name__)

# earth_model = joblib.load('earthquake_test.pkl')
# hurri_model = joblib.load('hurricane_test.pkl')
# app = Flask(__name__)
# CORS(app)

# @app.route("/")
# def hello_world():
#     return render_template('index.html')

# @app.route("/download")
# def download():
#     return render_template('download.html')


# @app.route("/detect", methods=['POST'])
# def detect():
#     print("detect route")
#     if request.method != "POST":
#         return "Invalid request method", 400

#     video = request.files.get('video')
    
#     if not video:
#         return "No video file provided", 400
#     print(video)
#     filename = secure_filename(video.filename)
#     video_path = os.path.join('static', filename)
#     print(video_path)
#     video.save(video_path)

#     output_filename = filename
#     output_path = os.path.join('static', output_filename)
#     print(output_path)
#     subprocess.run(['python', 'detect.py', '--source', video_path])
#     # subprocess.run(['python', 'detect.py', '--source', "static/input.mp4"])
#     return jsonify({"output_path": output_path})

# def earth_predicted_magnitude(latitude, longitude):
#     user_input = np.array([[latitude, longitude]])
#     prediction = earth_model.predict(user_input)
#     print(prediction)
#     return prediction[0]

# def predict_max_wind(latitude, longitude, moderate_wind_ne, moderate_wind_se, moderate_wind_sw, moderate_wind_nw, year, month, day):
#     input_data = np.array([[latitude, longitude, moderate_wind_ne, moderate_wind_se, moderate_wind_sw, moderate_wind_nw, year, month, day]])
#     prediction = hurri_model.predict(input_data)
#     return prediction[0]

# def flood_predict(month,value):
#     file_path = 'kerala.csv'
#     data = pd.read_csv(file_path)
#     correct_column_name = 'FLOODS'
#     flood_data = data[data[correct_column_name] == 'YES']
#     thresholds = flood_data.drop(['SUBDIVISION', 'YEAR', ' ANNUAL RAINFALL', 'FLOODS'], axis=1).max()

#     print("Threshold values for monthly rainfall above which a flood is likely to occur:")
#     def predict_flood(month , value , k=thresholds):
#         if value is not None and value > thresholds[month.upper()]:
#             return "Flood likely"
#         return "No flood"

#     prediction = predict_flood(month,value)
#     print(f'Prediction: {prediction}')
#     return prediction

# @app.route('/flood', methods=['POST'])
# def flood():
#     data=request.get_json()
#     month = data.get('month')
#     value=data.get('value')
#     print(month," ", value)
#     prediction=flood_predict(month,value)
#     return jsonify({"predicted_FLOOD": prediction})

# # HURRICANE ROUTE
# @app.route('/hurri', methods=['POST'])
# def hurri():
#     data=request.get_json()
#     latitude = data.get('latitude')
#     longitude = data.get('longitude')
#     moderate_wind_ne =data.get('moderate_wind_ne')
#     moderate_wind_se =data.get('moderate_wind_se')
#     moderate_wind_sw =data.get('moderate_wind_sw')
#     moderate_wind_nw =data.get('moderate_wind_nw')
#     year = data.get('year')
#     month = data.get('month')
#     day = data.get('day')
#     print(data)
#     predicted_max_wind1 = predict_max_wind(latitude, longitude, moderate_wind_ne, moderate_wind_se, moderate_wind_sw, moderate_wind_nw, year, month, day)
#     print(predicted_max_wind1)
#     return jsonify({"predicted_max_wind": predicted_max_wind1})

# @app.route('/earth', methods=['POST'])
# def earth():
#     data = request.get_json()
#     latitude=data.get('latitude')
#     longitude=data.get('longitude')
#     predicted_magnitude1 = earth_predicted_magnitude(latitude, longitude)
#     return jsonify({"predicted_magnitude": predicted_magnitude1})

# if __name__ == '__main__':
#     app.run(port=5001, debug=False)

from re import sub
from werkzeug.utils import secure_filename
import os
import subprocess
from flask import Flask, jsonify, request, render_template
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load ML models
earth_model = joblib.load('earthquake_test.pkl')
hurri_model = joblib.load('hurricane_test.pkl')

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/download")
def download():
    return render_template('download.html')

@app.route("/detect", methods=['POST'])
def detect():
    if request.method != "POST":
        return jsonify({"error": "Invalid request method"}), 400

    video = request.files.get('video')
    if not video:
        return jsonify({"error": "No video file provided"}), 400

    filename = secure_filename(video.filename)
    video_path = os.path.join('static', filename)
    video.save(video_path)

    # Run detection script
    subprocess.run(['python', 'detect.py', '--source', video_path])

    # Ensure output path exists before returning it
    output_path = os.path.join('static', filename)
    if os.path.exists(output_path):
        return jsonify({"output_path": output_path})
    else:
        return jsonify({"error": "Detection output not found"}), 500

def earth_predicted_magnitude(latitude, longitude):
    user_input = np.array([[latitude, longitude]])
    prediction = earth_model.predict(user_input)
    return float(prediction[0])

def predict_max_wind(latitude, longitude, moderate_wind_ne, moderate_wind_se, moderate_wind_sw, moderate_wind_nw, year, month, day):
    input_data = np.array([[latitude, longitude, moderate_wind_ne, moderate_wind_se, moderate_wind_sw, moderate_wind_nw, year, month, day]])
    prediction = hurri_model.predict(input_data)
    return float(prediction[0])

# Define flood prediction logic outside flood_predict()
def predict_flood(month, value, thresholds):
    if value is not None and value > thresholds[month.upper()]:
        return "Flood likely"
    return "No flood"

def flood_predict(month, value):
    file_path = 'kerala.csv'
    data = pd.read_csv(file_path)
    correct_column_name = 'FLOODS'
    flood_data = data[data[correct_column_name] == 'YES']
    thresholds = flood_data.drop(['SUBDIVISION', 'YEAR', ' ANNUAL RAINFALL', 'FLOODS'], axis=1).max()

    return predict_flood(month, value, thresholds)

@app.route('/flood', methods=['POST'])
def flood():
    data = request.get_json()
    month = data.get('month')
    value = data.get('value')

    if not month or value is None:
        return jsonify({"error": "Invalid input"}), 400

    prediction = flood_predict(month, value)
    return jsonify({"predicted_FLOOD": prediction})

@app.route('/hurri', methods=['POST'])
def hurri():
    data = request.get_json()
    required_fields = ["latitude", "longitude", "moderate_wind_ne", "moderate_wind_se", "moderate_wind_sw", "moderate_wind_nw", "year", "month", "day"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    predicted_max_wind1 = predict_max_wind(**data)
    return jsonify({"predicted_max_wind": predicted_max_wind1})

@app.route('/earth', methods=['POST'])
def earth():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    if latitude is None or longitude is None:
        return jsonify({"error": "Invalid input"}), 400

    predicted_magnitude1 = earth_predicted_magnitude(latitude, longitude)
    return jsonify({"predicted_magnitude": predicted_magnitude1})

if __name__ == '__main__':
    app.run(port=5001,host="0.0.0.0", debug=True)
