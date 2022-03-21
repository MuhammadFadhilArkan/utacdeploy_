from pickle import TRUE
from flask import Flask, request, jsonify, Response
import numpy as np
from pydantic import ValidationError
import prometheus_client
from datetime import datetime
from app.middleware import setup_metrics
from app.chemical_tin.hours3.data_structure import Ct3hours
from app.chemical_tin.hours24.data_structure import Ct24hours
from app.chemical_tin.hours168.data_structure import Ct168hours
from app.solder_thickness.hours3.data_structure import St3hours
from app.solder_thickness.hours24.data_structure import St24hours
from app.solder_thickness.hours168.data_structure import St168hours
from app.resources import RSC
from app.database import DB
from app.prometheus import PROM
from mlflow.tracking.client import MlflowClient
import mlflow
import os

app = Flask(__name__)

print("loading model")
rsc = RSC(is_first=True)
print("model loaded...")

print("creating database")
db = DB(host='mongo',port='27017',username='root',password='Secret')
print("database created")

print("creating mlflow client")
ip = os.environ['RETRAIN_IP']
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{ip}:9000"
mlflow.set_tracking_uri(f'http://{ip}:5000')
client = MlflowClient()
print("mlflow client created")

print("setting up prometheus endpoint")
prom = PROM()
setup_metrics(app)
@app.route('/metrics')
def metrics():
    CONTENT_TYPE_LATEST = str('text/plain; version=0.0.4; charset=utf-8')
    return Response(prometheus_client.generate_latest(), mimetype=CONTENT_TYPE_LATEST)
print("prometheus endpoint created")


#REST API
@app.route('/predict/ct/hours3', methods=['POST'])
def predict_ct_hours3():
    if request.method == 'POST':
        #get data
        context = request.get_json()
        data = []
        for i in range (len(rsc.features_ct3hours)):
            data.append(context[f"{rsc.features_ct3hours[i]}"])

        #normalize data
        data = np.array(data)
        data = data.reshape(1,-1)
        data = rsc.X_scaler_ct3hours.transform(data)
        data = data.flatten()

        #validate type and total amount of feature
        try:
            dummy = Ct3hours(mean_Converyer_Belt_Speed_m_min=data[0],
                            mean_Blower_Pressure_Bar=data[1],
                            mean_MatteTIn_Curent_Amp=data[2],
                            std_Converyer_Belt_Speed_m_min=data[3],
                            std_Blower_Pressure_Bar=data[4],
                            std_MatteTIn_Curent_Amp=data[5],
                            min_Converyer_Belt_Speed_m_min=data[6],
                            min_Blower_Pressure_Bar=data[7],
                            min_MatteTIn_Curent_Amp=data[8],
                            max_Blower_Pressure_Bar=data[9],
                            max_MatteTIn_Curent_Amp=data[10],
                            qntl1_Blower_Pressure_Bar=data[11],
                            qntl1_MatteTIn_Curent_Amp=data[12],
                            qntl3_Blower_Pressure_Bar=data[13],
                            qntl3_MatteTIn_Curent_Amp=data[14],
                            median_Converyer_Belt_Speed_m_min=data[15],
                            median_Blower_Pressure_Bar=data[16],
                            median_MatteTIn_Curent_Amp=data[17],
                            rows=data[18],
                            dow=data[19]
                            )
        except ValidationError as e:
            return e.json()

        #validate value is acceptable (between 0 and 1)
        acceptable = (np.max(data) <= 1.001) and (np.min(data) >= -0.001) 

        #predict
        data = np.array(data,dtype=np.float32)
        prediction = rsc.model_ct3hours.run(["dense_4"], {"dense_input": data.reshape(1,-1)})
        prediction = rsc.Y_scaler_ct3hours.inverse_transform(prediction[0])
        prediction = prediction[0][0]

        #forprometheus
        prom.REQUEST_PREDICTION_CT3hours.labels('chemical tin 3 hours ahead', request.method, request.path).set(prediction)

        #create item for database
        mydata = {  "mean_Converyer_Belt_Speed_m_min":str(data[0]),
                    "mean_Blower_Pressure_Bar":str(data[1]),
                    "mean_MatteTIn_Curent_Amp":str(data[2]),
                    "std_Converyer_Belt_Speed_m_min":str(data[3]),
                    "std_Blower_Pressure_Bar":str(data[4]),
                    "std_MatteTIn_Curent_Amp":str(data[5]),
                    "min_Converyer_Belt_Speed_m_min":str(data[6]),
                    "min_Blower_Pressure_Bar":str(data[7]),
                    "min_MatteTIn_Curent_Amp":str(data[8]),
                    "max_Blower_Pressure_Bar":str(data[9]),
                    "max_MatteTIn_Curent_Amp":str(data[10]),
                    "qntl1_Blower_Pressure_Bar":str(data[11]),
                    "qntl1_MatteTIn_Curent_Amp":str(data[12]),
                    "qntl3_Blower_Pressure_Bar":str(data[13]),
                    "qntl3_MatteTIn_Curent_Amp":str(data[14]),
                    "median_Converyer_Belt_Speed_m_min":str(data[15]),
                    "median_Blower_Pressure_Bar":str(data[16]),
                    "median_MatteTIn_Curent_Amp":str(data[17]),
                    "rows":str(data[18]),
                    "dow":str(data[19]),
                    "prediction":str(prediction),
                    "datetime":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                }

        #insert to database
        db.Ct3hours_col.insert_one(mydata)

        #return prediction
        try:
            return jsonify({'Prediction': str(prediction),"Info":" "}) if acceptable else jsonify({'Prediction': str(prediction),"Info":"Data is outside Training distribution So the prediction might not be accurate"})
        except:
            return 'There is an issue'

@app.route('/predict/ct/hours24', methods=['POST'])
def predict_ct_hours24():
    if request.method == 'POST':
        #get data
        context = request.get_json()
        data = []
        for i in range (len(rsc.features_ct24hours)):
            data.append(context[f"{rsc.features_ct24hours[i]}"])

        #normalize data
        data = np.array(data)
        data = data.reshape(1,-1)
        data = rsc.X_scaler_ct24hours.transform(data)
        data = data.flatten()

        #validate type and total amount of feature
        try:
            dummy = Ct24hours(  mean_Converyer_Belt_Speed_m_min=data[0],
                                mean_Blower_Pressure_Bar=data[1],
                                mean_MatteTIn_Curent_Amp=data[2],
                                std_Converyer_Belt_Speed_m_min=data[3],
                                std_Blower_Pressure_Bar=data[4],
                                std_MatteTIn_Curent_Amp=data[5],
                                min_Converyer_Belt_Speed_m_min=data[6],
                                min_Blower_Pressure_Bar=data[7],
                                min_MatteTIn_Curent_Amp=data[8],
                                max_Converyer_Belt_Speed_m_min=data[9],
                                max_Blower_Pressure_Bar=data[10],
                                qntl1_Converyer_Belt_Speed_m_min=data[11],
                                qntl1_Blower_Pressure_Bar=data[12],
                                qntl1_MatteTIn_Curent_Amp=data[13],
                                qntl3_Converyer_Belt_Speed_m_min=data[14],
                                qntl3_Blower_Pressure_Bar=data[15],
                                median_Converyer_Belt_Speed_m_min=data[16],
                                median_Blower_Pressure_Bar=data[17],
                                rows=data[18],
                                dow=data[19]
                            )
        except ValidationError as e:
            return e.json()

        #validate value is acceptable (between 0 and 1)
        acceptable = (np.max(data) <= 1.001) and (np.min(data) >= -0.001) 

        #predict
        data = np.array(data,dtype=np.float32)
        prediction = rsc.model_ct24hours.run(["dense_4"], {"dense_input": data.reshape(1,-1)})
        prediction = rsc.Y_scaler_ct24hours.inverse_transform(prediction[0])
        prediction = prediction[0][0]

        #forprometheus
        prom.REQUEST_PREDICTION_CT24hours.labels('chemical tin 24 hours ahead', request.method, request.path).set(prediction)

        #create item for database
        mydata = {  "mean_Converyer_Belt_Speed_m_min":str(data[0]),
                    "mean_Blower_Pressure_Bar":str(data[1]),
                    "mean_MatteTIn_Curent_Amp":str(data[2]),
                    "std_Converyer_Belt_Speed_m_min":str(data[3]),
                    "std_Blower_Pressure_Bar":str(data[4]),
                    "std_MatteTIn_Curent_Amp":str(data[5]),
                    "min_Converyer_Belt_Speed_m_min":str(data[6]),
                    "min_Blower_Pressure_Bar":str(data[7]),
                    "min_MatteTIn_Curent_Amp":str(data[8]),
                    "max_Converyer_Belt_Speed_m_min":str(data[9]),
                    "max_Blower_Pressure_Bar":str(data[10]),
                    "qntl1_Converyer_Belt_Speed_m_min":str(data[11]),
                    "qntl1_Blower_Pressure_Bar":str(data[12]),
                    "qntl1_MatteTIn_Curent_Amp":str(data[13]),
                    "qntl3_Converyer_Belt_Speed_m_min":str(data[14]),
                    "qntl3_Blower_Pressure_Bar":str(data[15]),
                    "median_Converyer_Belt_Speed_m_min":str(data[16]),
                    "median_Blower_Pressure_Bar":str(data[17]),
                    "rows":str(data[18]),
                    "dow":str(data[19]),
                    "prediction":str(prediction),
                    "datetime":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                }

        #insert to database
        db.Ct24hours_col.insert_one(mydata)

        #return prediction
        try:
            return jsonify({'Prediction': str(prediction),"Info":" "}) if acceptable else jsonify({'Prediction': str(prediction),"Info":"Data is outside Training distribution So the prediction might not be accurate"})
        except:
            return 'There is an issue'

@app.route('/predict/ct/hours168', methods=['POST'])
def predict_ct_hours168():
    if request.method == 'POST':
        #get data
        context = request.get_json()
        data = []
        for i in range (len(rsc.features_ct168hours)):
            data.append(context[f"{rsc.features_ct168hours[i]}"])

        #normalize data
        data = np.array(data)
        data = data.reshape(1,-1)
        data = rsc.X_scaler_ct168hours.transform(data)
        data = data.flatten()

        #validate type and total amount of feature
        try:
            dummy = Ct168hours( mean_Converyer_Belt_Speed_m_min=data[0],
                                mean_Blower_Pressure_Bar=data[1],
                                mean_MatteTIn_Curent_Amp=data[2],
                                std_Converyer_Belt_Speed_m_min=data[3],
                                std_Blower_Pressure_Bar=data[4],
                                std_MatteTIn_Curent_Amp=data[5],
                                min_Converyer_Belt_Speed_m_min=data[6],
                                min_Blower_Pressure_Bar=data[7],
                                max_Converyer_Belt_Speed_m_min=data[8],
                                max_Blower_Pressure_Bar=data[9],
                                max_MatteTIn_Curent_Amp=data[10],
                                qntl1_Converyer_Belt_Speed_m_min=data[11],
                                qntl1_Blower_Pressure_Bar=data[12],
                                qntl3_Converyer_Belt_Speed_m_min=data[13],
                                qntl3_Blower_Pressure_Bar=data[14],
                                qntl3_MatteTIn_Curent_Amp=data[15],
                                median_Converyer_Belt_Speed_m_min=data[16],
                                median_Blower_Pressure_Bar=data[17],
                                rows=data[18],
                                dow=data[19]
                            )
        except ValidationError as e:
            return e.json()

        #validate value is acceptable (between 0 and 1)
        acceptable = (np.max(data) <= 1.001) and (np.min(data) >= -0.001) 

        #predict
        data = np.array(data,dtype=np.float32)
        prediction = rsc.model_ct168hours.run(["dense_4"], {"dense_input": data.reshape(1,-1)})
        prediction = rsc.Y_scaler_ct168hours.inverse_transform(prediction[0])
        prediction = prediction[0][0]

        #forprometheus
        prom.REQUEST_PREDICTION_CT168hours.labels('chemical tin 168 hours ahead', request.method, request.path).set(prediction)

        #create item for database
        mydata = {  "mean_Converyer_Belt_Speed_m_min":str(data[0]),
                    "mean_Blower_Pressure_Bar":str(data[1]),
                    "mean_MatteTIn_Curent_Amp":str(data[2]),
                    "std_Converyer_Belt_Speed_m_min":str(data[3]),
                    "std_Blower_Pressure_Bar":str(data[4]),
                    "std_MatteTIn_Curent_Amp":str(data[5]),
                    "min_Converyer_Belt_Speed_m_min":str(data[6]),
                    "min_Blower_Pressure_Bar":str(data[7]),
                    "max_Converyer_Belt_Speed_m_min":str(data[8]),
                    "max_Blower_Pressure_Bar":str(data[9]),
                    "max_MatteTIn_Curent_Amp":str(data[10]),
                    "qntl1_Converyer_Belt_Speed_m_min":str(data[11]),
                    "qntl1_Blower_Pressure_Bar":str(data[12]),
                    "qntl3_Converyer_Belt_Speed_m_min":str(data[13]),
                    "qntl3_Blower_Pressure_Bar":str(data[14]),
                    "qntl3_MatteTIn_Curent_Amp":str(data[15]),
                    "median_Converyer_Belt_Speed_m_min":str(data[16]),
                    "median_Blower_Pressure_Bar":str(data[17]),
                    "rows":str(data[18]),
                    "dow":str(data[19]),
                    "prediction":str(prediction),
                    "datetime":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                }

        #insert to database
        db.Ct168hours_col.insert_one(mydata)

        #return prediction
        try:
            return jsonify({'Prediction': str(prediction),"Info":" "}) if acceptable else jsonify({'Prediction': str(prediction),"Info":"Data is outside Training distribution So the prediction might not be accurate"})
        except:
            return 'There is an issue'

@app.route('/predict/st/hours3', methods=['POST'])
def predict_st_hours3():
    if request.method == 'POST':
        #get data
        context = request.get_json()
        data = []
        for i in range (len(rsc.features_st3hours)):
            data.append(context[f"{rsc.features_st3hours[i]}"])

        #normalize data
        data = np.array(data)
        data = data.reshape(1,-1)
        data = rsc.X_scaler_st3hours.transform(data)
        data = data.flatten()

        #validate type and total amount of feature
        try:
            dummy = St3hours(   mean_Converyer_Belt_Speed_m_min=data[0],
                                mean_Blower_Pressure_Bar=data[1],
                                mean_MatteTIn_Curent_Amp=data[2],
                                std_Converyer_Belt_Speed_m_min=data[3],
                                std_MatteTIn_Curent_Amp=data[4],
                                min_Converyer_Belt_Speed_m_min=data[5],
                                min_Blower_Pressure_Bar=data[6],
                                max_Converyer_Belt_Speed_m_min=data[7],
                                max_Blower_Pressure_Bar=data[8],
                                max_MatteTIn_Curent_Amp=data[9],
                                qntl1_Converyer_Belt_Speed_m_min=data[10],
                                qntl1_Blower_Pressure_Bar=data[11],
                                qntl3_Converyer_Belt_Speed_m_min=data[12],
                                qntl3_Blower_Pressure_Bar=data[13],
                                median_Converyer_Belt_Speed_m_min=data[14],
                                median_Blower_Pressure_Bar=data[15],
                                median_MatteTIn_Curent_Amp=data[16],
                                rows=data[17],
                                dow=data[18],
                                type=data[19],
                            )
        except ValidationError as e:
            return e.json()

        #validate value is acceptable (between 0 and 1)
        acceptable = (np.max(data) <= 1.001) and (np.min(data) >= -0.001) 

        #predict
        data = np.array(data,dtype=np.float32)
        prediction = rsc.model_st3hours.run(["dense_4"], {"dense_input": data.reshape(1,-1)})
        prediction = rsc.Y_scaler_st3hours.inverse_transform(prediction[0])
        prediction = prediction[0][0]

        #forprometheus
        prom.REQUEST_PREDICTION_ST3hours.labels('solder thickness 3 hours ahead', request.method, request.path).set(prediction)

        #create item for database
        mydata = {  "mean_Converyer_Belt_Speed_m_min":str(data[0]),
                    "mean_Blower_Pressure_Bar":str(data[1]),
                    "mean_MatteTIn_Curent_Amp":str(data[2]),
                    "std_Converyer_Belt_Speed_m_min":str(data[3]),
                    "std_MatteTIn_Curent_Amp":str(data[4]),
                    "min_Converyer_Belt_Speed_m_min":str(data[5]),
                    "min_Blower_Pressure_Bar":str(data[6]),
                    "max_Converyer_Belt_Speed_m_min":str(data[7]),
                    "max_Blower_Pressure_Bar":str(data[8]),
                    "max_MatteTIn_Curent_Amp":str(data[9]),
                    "qntl1_Converyer_Belt_Speed_m_min":str(data[10]),
                    "qntl1_Blower_Pressure_Bar":str(data[11]),
                    "qntl3_Converyer_Belt_Speed_m_min":str(data[12]),
                    "qntl3_Blower_Pressure_Bar":str(data[13]),
                    "median_Converyer_Belt_Speed_m_min":str(data[14]),
                    "median_Blower_Pressure_Bar":str(data[15]),
                    "median_MatteTIn_Curent_Amp":str(data[16]),
                    "rows":str(data[17]),
                    "dow":str(data[18]),
                    "type":str(data[19]),
                    "prediction":str(prediction),
                    "datetime":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                }

        #insert to database
        db.St3hours_col.insert_one(mydata)

        #return prediction
        try:
            return jsonify({'Prediction': str(prediction),"Info":" "}) if acceptable else jsonify({'Prediction': str(prediction),"Info":"Data is outside Training distribution So the prediction might not be accurate"})
        except:
            return 'There is an issue'

@app.route('/predict/st/hours24', methods=['POST'])
def predict_st_hours24():
    if request.method == 'POST':
        #get data
        context = request.get_json()
        data = []
        for i in range (len(rsc.features_st24hours)):
            data.append(context[f"{rsc.features_st24hours[i]}"])

        #normalize data
        data = np.array(data)
        data = data.reshape(1,-1)
        data = rsc.X_scaler_st24hours.transform(data)
        data = data.flatten()

        #validate type and total amount of feature
        try:
            dummy = St24hours(  mean_Converyer_Belt_Speed_m_min=data[0],
                                mean_Blower_Pressure_Bar=data[1],
                                mean_MatteTIn_Curent_Amp=data[2],
                                std_Converyer_Belt_Speed_m_min=data[3],
                                std_Blower_Pressure_Bar=data[4],
                                min_Blower_Pressure_Bar=data[5],
                                max_Converyer_Belt_Speed_m_min=data[6],
                                max_Blower_Pressure_Bar=data[7],
                                qntl1_Converyer_Belt_Speed_m_min=data[8],
                                qntl1_Blower_Pressure_Bar=data[9],
                                qntl1_MatteTIn_Curent_Amp=data[10],
                                qntl3_Converyer_Belt_Speed_m_min=data[11],
                                qntl3_Blower_Pressure_Bar=data[12],
                                median_Converyer_Belt_Speed_m_min=data[13],
                                median_Blower_Pressure_Bar=data[14],
                                median_MatteTIn_Curent_Amp=data[15],
                                rows=data[16],
                                dow=data[17],
                                hod=data[18],
                                type=data[19],
                            )
        except ValidationError as e:
            return e.json()

        #validate value is acceptable (between 0 and 1)
        acceptable = (np.max(data) <= 1.001) and (np.min(data) >= -0.001) 

        #predict
        data = np.array(data,dtype=np.float32)
        prediction = rsc.model_st24hours.run(["dense_4"], {"dense_input": data.reshape(1,-1)})
        prediction = rsc.Y_scaler_st24hours.inverse_transform(prediction[0])
        prediction = prediction[0][0]

        #forprometheus
        prom.REQUEST_PREDICTION_ST24hours.labels('solder thickness 24 hours ahead', request.method, request.path).set(prediction)

        #create item for database
        mydata = {  "mean_Converyer_Belt_Speed_m_min":str(data[0]),
                    "mean_Blower_Pressure_Bar":str(data[1]),
                    "mean_MatteTIn_Curent_Amp":str(data[2]),
                    "std_Converyer_Belt_Speed_m_min":str(data[3]),
                    "std_Blower_Pressure_Bar":str(data[4]),
                    "min_Blower_Pressure_Bar":str(data[5]),
                    "max_Converyer_Belt_Speed_m_min":str(data[6]),
                    "max_Blower_Pressure_Bar":str(data[7]),
                    "qntl1_Converyer_Belt_Speed_m_min":str(data[8]),
                    "qntl1_Blower_Pressure_Bar":str(data[9]),
                    "qntl1_MatteTIn_Curent_Amp":str(data[10]),
                    "qntl3_Converyer_Belt_Speed_m_min":str(data[11]),
                    "qntl3_Blower_Pressure_Bar":str(data[12]),
                    "median_Converyer_Belt_Speed_m_min":str(data[13]),
                    "median_Blower_Pressure_Bar":str(data[14]),
                    "median_MatteTIn_Curent_Amp":str(data[15]),
                    "rows":str(data[16]),
                    "dow":str(data[17]),
                    "hod":str(data[18]),
                    "type":str(data[19]),
                    "prediction":str(prediction),
                    "datetime":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                }

        #insert to database
        db.St24hours_col.insert_one(mydata)

        #return prediction
        try:
            return jsonify({'Prediction': str(prediction),"Info":" "}) if acceptable else jsonify({'Prediction': str(prediction),"Info":"Data is outside Training distribution So the prediction might not be accurate"})
        except:
            return 'There is an issue'

@app.route('/predict/st/hours168', methods=['POST'])
def predict_st_hours168():
    if request.method == 'POST':
        #get data
        context = request.get_json()
        data = []
        for i in range (len(rsc.features_st168hours)):
            data.append(context[f"{rsc.features_st168hours[i]}"])

        #normalize data
        data = np.array(data)
        data = data.reshape(1,-1)
        data = rsc.X_scaler_st168hours.transform(data)
        data = data.flatten()

        #validate type and total amount of feature
        try:
            dummy = St168hours( mean_Converyer_Belt_Speed_m_min=data[0],
                                mean_Blower_Pressure_Bar=data[1],
                                mean_MatteTIn_Curent_Amp=data[2],
                                std_Converyer_Belt_Speed_m_min=data[3],
                                std_Blower_Pressure_Bar=data[4],
                                std_MatteTIn_Curent_Amp=data[5],
                                min_Converyer_Belt_Speed_m_min=data[6],
                                min_Blower_Pressure_Bar=data[7],
                                max_Converyer_Belt_Speed_m_min=data[8],
                                max_Blower_Pressure_Bar=data[9],
                                max_MatteTIn_Curent_Amp=data[10],
                                qntl1_Blower_Pressure_Bar=data[11],
                                qntl1_MatteTIn_Curent_Amp=data[12],
                                qntl3_Blower_Pressure_Bar=data[13],
                                median_Blower_Pressure_Bar=data[14],
                                median_MatteTIn_Curent_Amp=data[15],
                                rows=data[16],
                                dow=data[17],
                                hod=data[18],
                                type=data[19],
                            )
        except ValidationError as e:
            return e.json()

        #validate value is acceptable (between 0 and 1)
        acceptable = (np.max(data) <= 1.001) and (np.min(data) >= -0.001) 

        #predict
        data = np.array(data,dtype=np.float32)
        prediction = rsc.model_st168hours.run(["dense_4"], {"dense_input": data.reshape(1,-1)})
        prediction = rsc.Y_scaler_st168hours.inverse_transform(prediction[0])
        prediction = prediction[0][0]

        #forprometheus
        prom.REQUEST_PREDICTION_ST168hours.labels('solder thickness 168 hours ahead', request.method, request.path).set(prediction)

        #create item for database
        mydata = {  "mean_Converyer_Belt_Speed_m_min":str(data[0]),
                    "mean_Blower_Pressure_Bar":str(data[1]),
                    "mean_MatteTIn_Curent_Amp":str(data[2]),
                    "std_Converyer_Belt_Speed_m_min":str(data[3]),
                    "std_Blower_Pressure_Bar":str(data[4]),
                    "std_MatteTIn_Curent_Amp":str(data[5]),
                    "min_Converyer_Belt_Speed_m_min":str(data[6]),
                    "min_Blower_Pressure_Bar":str(data[7]),
                    "max_Converyer_Belt_Speed_m_min":str(data[8]),
                    "max_Blower_Pressure_Bar":str(data[9]),
                    "max_MatteTIn_Curent_Amp":str(data[10]),
                    "qntl1_Blower_Pressure_Bar":str(data[11]),
                    "qntl1_MatteTIn_Curent_Amp":str(data[12]),
                    "qntl3_Blower_Pressure_Bar":str(data[13]),
                    "median_Blower_Pressure_Bar":str(data[14]),
                    "median_MatteTIn_Curent_Amp":str(data[15]),
                    "rows":str(data[16]),
                    "dow":str(data[17]),
                    "hod":str(data[18]),
                    "type":str(data[19]),
                    "prediction":str(prediction),
                    "datetime":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                }

        #insert to database
        db.St168hours_col.insert_one(mydata)

        #return prediction
        try:
            return jsonify({'Prediction': str(prediction),"Info":" "}) if acceptable else jsonify({'Prediction': str(prediction),"Info":"Data is outside Training distribution So the prediction might not be accurate"})
        except:
            return 'There is an issue'

@app.route('/gettop5', methods=['GET'])
def gettop5():

    top5_ct3hours = {}
    top5_ct24hours = {}
    top5_ct168hours = {}
    top5_st3hours = {}
    top5_st24hours = {}
    top5_st168hours = {}

    top5_ct3hours_list = []
    top5_ct24hours_list = []
    top5_ct168hours_list = []
    top5_st3hours_list = []
    top5_st24hours_list = []
    top5_st168hours_list = []

    for i in range (5):
        top5_ct3hours[f'{rsc.top5_ct3hours[i][0]}'] = rsc.top5_ct3hours[i][1]
        top5_ct24hours[f'{rsc.top5_ct24hours[i][0]}'] = rsc.top5_ct24hours[i][1]
        top5_ct168hours[f'{rsc.top5_ct168hours[i][0]}'] = rsc.top5_ct168hours[i][1]
        top5_st3hours[f'{rsc.top5_st3hours[i][0]}'] = rsc.top5_st3hours[i][1]
        top5_st24hours[f'{rsc.top5_st24hours[i][0]}'] = rsc.top5_st24hours[i][1]
        top5_st168hours[f'{rsc.top5_st168hours[i][0]}'] = rsc.top5_st168hours[i][1]

    top5_ct3hours_list.append(top5_ct3hours)
    top5_ct24hours_list.append(top5_ct24hours)
    top5_ct168hours_list.append(top5_ct168hours)
    top5_st3hours_list.append(top5_st3hours)
    top5_st24hours_list.append(top5_st24hours)
    top5_st168hours_list.append(top5_st168hours)

    try:
        return jsonify({'top5_ct3hours': top5_ct3hours_list,
                        'top5_ct24hours': top5_ct24hours_list,
                        'top5_ct168hours': top5_ct168hours_list,
                        'top5_st3hours': top5_st3hours_list,
                        'top5_st24hours': top5_st24hours_list,
                        'top5_st168hours': top5_st168hours_list
                        })
    except:
        return 'There is an issue'

@app.route('/update_model', methods=['POST'])
def update_model():

    context = request.get_json()
    task = context['task']
    hours = context['hours']

    print("reinit resource")
    from app.resources import RSC
    global rsc
    rsc = RSC()
    print("new resource is initialized")

    latest_version_info = client.get_latest_versions(f'{task}_{hours}', stages=["Production"])
    latest_production_version = int(latest_version_info[0].version)

    if task=="chemical_tin":
        if hours==3:
            prom.MODEL_VERSION_CT3hours.labels('chemical tin 3 hours ahead', request.method, request.path).set(latest_production_version)
            prom.MODEL_RETRAIN_CT3hours.labels('chemical tin 3 hours ahead', request.method, request.path).inc()
        elif hours==24:
            prom.MODEL_VERSION_CT24hours.labels('chemical tin 24 hours ahead', request.method, request.path).set(latest_production_version)
            prom.MODEL_RETRAIN_CT24hours.labels('chemical tin 24 hours ahead', request.method, request.path).inc()
        elif hours==168:
            prom.MODEL_VERSION_CT168hours.labels('chemical tin 168 hours ahead', request.method, request.path).set(latest_production_version)
            prom.MODEL_RETRAIN_CT168hours.labels('chemical tin 168 hours ahead', request.method, request.path).inc()
    elif task=="solder_thickness":
        if hours==3:
            prom.MODEL_VERSION_ST3hours.labels('solder thickness 3 hours ahead', request.method, request.path).set(latest_production_version)
            prom.MODEL_RETRAIN_ST3hours.labels('solder thickness 3 hours ahead', request.method, request.path).inc()
        elif hours==24:
            prom.MODEL_VERSION_ST24hours.labels('solder thickness 24 hours ahead', request.method, request.path).set(latest_production_version)
            prom.MODEL_RETRAIN_ST24hours.labels('solder thickness 24 hours ahead', request.method, request.path).inc()
        elif hours==168:
            prom.MODEL_VERSION_ST168hours.labels('solder thickness 168 hours ahead', request.method, request.path).set(latest_production_version)
            prom.MODEL_RETRAIN_ST168hours.labels('solder thickness 168 hours ahead', request.method, request.path).inc()

    try:
        return 'Success'
    except:
        return 'There is an issue'

@app.route('/update_training_status', methods=['POST'])
def update_training_status():

    context = request.get_json()
    task = context['task']
    hours = context['hours']
    status = context['status']

    if task=="chemical_tin":
        if hours==3:
            prom.MODEL_STATUS_CT3hours.state(f'{status}')
        if hours==24:
            prom.MODEL_STATUS_CT24hours.state(f'{status}')
        if hours==168:
            prom.MODEL_STATUS_CT168hours.state(f'{status}')

    if task=="solder_thickness":
        if hours==3:
            prom.MODEL_STATUS_ST3hours.state(f'{status}')   
        if hours==24:
            prom.MODEL_STATUS_ST24hours.state(f'{status}')
        if hours==168:
            prom.MODEL_STATUS_ST168hours.state(f'{status}')    

    try:
        return 'Success'
    except:
        return 'There is an issue'

@app.route('/add_gtruth', methods=['POST'])
def add_gtruth():

    context = request.get_json()
    task = context['task']
    gtruth = context['gtruth']

    #insert to database
    if task=="chemical_tin":
        db.ct_gtruth.insert_one(gtruth)
    elif task=="solder_thickness":
        db.st_gtruth.insert_one(gtruth)

    try:
        return 'Success'
    except:
        return 'There is an issue'

if __name__ == "__main__":
    app.run(host="127.0.0.1",debug=False)
