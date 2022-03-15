from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import pandas as pd
import mlflow
import mlflow.tensorflow
from sklearn import metrics
import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import onnx
import os
import tf2onnx
import onnxruntime as rt
from pickle import dump, load
from retrain_app.minio_client import MINIO

def create_model_simpleNN(shape,shape_target):
    tf.keras.backend.clear_session()

    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=4096,
                          input_shape=[shape[1]], 
                          activation="relu"),
    tf.keras.layers.Dense(2048, 
                          activation="relu",
                          #activity_regularizer=tf.keras.regularizers.L2(0.01)
                          ),
    tf.keras.layers.Dense(1024, 
                          activation="relu",
                          #activity_regularizer=tf.keras.regularizers.L2(0.01)
                          ),  
    tf.keras.layers.Dense(512, 
                          activation="relu",
                          #activity_regularizer=tf.keras.regularizers.L2(0.01)
                          ),            
    #tf.keras.layers.Dropout(0.5),                     
    tf.keras.layers.Dense(shape_target, 
                          #activity_regularizer=tf.keras.regularizers.L2(0.01)
                          )
    ])

    model.summary()
    return model

def evaluation_metrics_func(y_true, y_pred,max,min):

    Normalized_MAE = metrics.mean_absolute_error(y_true, y_pred)/(max-min)
    Normalized_RMSE = np.sqrt(metrics.mean_squared_error(y_true, y_pred))/(max-min)
    MAE = metrics.mean_absolute_error(y_true, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    MAPE = np.sqrt(metrics.mean_absolute_percentage_error(y_true, y_pred))

    return Normalized_MAE,Normalized_RMSE,MAE,RMSE,MAPE

def evaluate_model(model,X_test,y_test,y,Y_scaler):

    y_Inverse = Y_scaler.inverse_transform(y)
    pred = model.predict(X_test)
    pred = pred.reshape(-1,1)
    pred_Inverse = Y_scaler.inverse_transform(pred)
    y_train_Inverse = Y_scaler.inverse_transform(y_test)
    Normalized_MAE,Normalized_RMSE,MAE,RMSE,MAPE = evaluation_metrics_func(y_train_Inverse,pred_Inverse,np.max(y_Inverse),np.min(y_Inverse))

    return Normalized_MAE,Normalized_RMSE,MAE,RMSE,MAPE

def evaluate_model_onnx(model,X_test,y_test,y,Y_scaler):

    y_Inverse = Y_scaler.inverse_transform(y)
    pred = []
    for data in X_test:
        prd = model.run(["dense_4"], {"dense_input": data.reshape(1,-1)})
        pred.append(prd)
    pred = np.array(pred)
    pred = pred.reshape(-1,1)
    pred_Inverse = Y_scaler.inverse_transform(pred)
    y_train_Inverse = Y_scaler.inverse_transform(y_test)
    Normalized_MAE,Normalized_RMSE,MAE,RMSE,MAPE = evaluation_metrics_func(y_train_Inverse,pred_Inverse,np.max(y_Inverse),np.min(y_Inverse))

    return Normalized_MAE,Normalized_RMSE,MAE,RMSE,MAPE

def wait_until_ready(model_name, model_version):
  client = MlflowClient()
  for _ in range(10):
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)

def register_model(model_name,run_id):
    artifact_path = "model/data/model"
    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    wait_until_ready(model_details.name, model_details.version)

    return model_details

def change_model_state(state,model_details):

    client = MlflowClient()
    client.transition_model_version_stage(name=model_details.name,
                                          version=model_details.version,
                                          stage=f'{state}',
                                          )

def change_model_state_2(state,model_name,model_version):

    client = MlflowClient()
    client.transition_model_version_stage(name=model_name,
                                          version=model_version,
                                          stage=f'{state}',
                                          )

def use_retrained_model(state,task,hours,run_id):
    model_name = f'{task}_{hours}'
    model_details = register_model(model_name,run_id)
    change_model_state(state,model_details)

def model_retrain(X,y,Y_scaler,X_Scaler,task,hours,epoch):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    batch_size = 1000
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_data = train_data.cache().batch(batch_size)
    val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_data = val_data.batch(batch_size)

    shape = X_train.shape[-2:]
    shape_target = y_train.shape[-1]

    ip = os.environ['RETRAIN_IP']
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{ip}:9000"

    mlflow.set_tracking_uri(f'http://{ip}:5000')
    mlflow.set_experiment(f"{task}_{hours}")

    with mlflow.start_run():
        mlflow.tensorflow.autolog()

        global retraining_info
        retraining_info = 'is training'

        #model = create_model_simpleNN(shape,shape_target)
        model = tf.keras.models.load_model(f"/var/www/app/{task}/hours{hours}/model.h5")
        optimizer = tf.keras.optimizers.Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        model.compile(loss=tf.keras.losses.Huber(),
                    optimizer=optimizer,
                    metrics=["mse"])
        history = model.fit(train_data,epochs=epoch,validation_data=val_data,verbose=1)
        Normalized_MAE,Normalized_RMSE,MAE,RMSE,MAPE = evaluate_model(model,X_test,y_test,y,Y_scaler)
        mlflow.log_metric("Normalized_MAE", Normalized_MAE)
        mlflow.log_metric("Normalized_RMSE", Normalized_RMSE)
        mlflow.log_metric("MAE", MAE)
        mlflow.log_metric("RMSE", RMSE)
        mlflow.log_metric("MAPE", MAPE)
        mlflow.log_artifact(f'/var/www/app/X_scaler.pkl')
        mlflow.log_artifact(f'/var/www/app/y_scaler.pkl')
        run_id = mlflow.active_run().info.run_id
        retraining_info = 'no training task'

        output_path = f"/var/www/app/{task}/hours{hours}/model.onnx"
        onnx_model, _ = tf2onnx.convert.from_keras(model)
        mlflow.onnx.log_model(onnx_model, "onnx_model")
        
        model_latest = rt.InferenceSession(f"/var/www/app/{task}/hours{hours}/model.onnx")
        Normalized_MAE_latest,Normalized_RMSE_latest,MAE_latest,RMSE_latest,MAPE_latest = evaluate_model_onnx(model_latest,X_test,y_test,y,Y_scaler)
        mlflow.log_metric("RMSE latest", RMSE_latest)

        if RMSE_latest>=RMSE:
            onnx.save(onnx_model, output_path)
            model.save(f"/var/www/app/{task}/hours{hours}/model.h5")
            dump(X_Scaler, open(f'/var/www/app/{task}/hours{hours}/X_scaler.pkl', 'wb'))
            dump(Y_scaler, open(f'/var/www/app/{task}/hours{hours}/y_scaler.pkl', 'wb'))        

            state = 'Archived'
            client = MlflowClient()
            latest_version_info = client.get_latest_versions(f'{task}_{hours}', stages=["Production"])
            latest_production_version = int(latest_version_info[0].version)
            change_model_state_2(state,f"{task}_{hours}",latest_production_version)

            state='Production'
            use_retrained_model(state,task,hours,run_id)

        else:
            state='Archived'
            use_retrained_model(state,task,hours,run_id)

        return True

def change_production_model_to(task,hours,model_version):

    name2id = {"chemical_tin_3":1,
               "chemical_tin_24":2,
               "chemical_tin_168":3,
               "solder_thickness_3":4,
               "solder_thickness_24":5,
               "solder_thickness_168":6
               }

    ip = os.environ['RETRAIN_IP']
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{ip}:9000"

    mlflow.set_tracking_uri(f'http://{ip}:5000')

    client = MlflowClient()
    model_name = f"{task}_{hours}"
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    run_id = model_version_details.run_id

    print("creating minio client")
    minio = MINIO()
    minioClient = minio.minioClient
    print("minio client created")

    exp_id = name2id[model_name]

    minioClient.fget_object('mlflow',
                                f"{exp_id}/{run_id}/artifacts/X_scaler.pkl",
                                f'/var/www/app/{task}/hours{hours}/X_scaler.pkl'
                                )
    minioClient.fget_object('mlflow',
                                f"{exp_id}/{run_id}/artifacts/y_scaler.pkl",
                                f'/var/www/app/{task}/hours{hours}/y_scaler.pkl'
                                )
    minioClient.fget_object('mlflow',
                                f"{exp_id}/{run_id}/artifacts/onnx_model/model.onnx",
                                f'/var/www/app/{task}/hours{hours}/model.onnx'
                                )

    state = 'Archived'
    latest_version_info = client.get_latest_versions(f'{task}_{hours}', stages=["Production"])
    latest_production_version = int(latest_version_info[0].version)
    change_model_state_2(state,model_name,latest_production_version)

    state = 'Production'
    wait_until_ready(model_version_details.name, model_version_details.version)
    change_model_state(state,model_version_details)

    return True