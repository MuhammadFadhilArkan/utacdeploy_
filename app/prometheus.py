from prometheus_client import Gauge, Counter, Enum

class PROM():

    def __init__(self):

        #Prediction Info
        self.REQUEST_PREDICTION_CT3hours = Gauge(
            'request_prediction_ct3hours', 'Prediction Result',
            ['app_name', 'method', 'endpoint']
        )
        self.REQUEST_PREDICTION_CT24hours = Gauge(
            'request_prediction_ct24hours', 'Prediction Result',
            ['app_name', 'method', 'endpoint']
        )
        self.REQUEST_PREDICTION_CT168hours = Gauge(
            'request_prediction_ct168hours', 'Prediction Result',
            ['app_name', 'method', 'endpoint']
        )

        self.REQUEST_PREDICTION_ST3hours = Gauge(
            'request_prediction_st3hours', 'Prediction Result',
            ['app_name', 'method', 'endpoint']
        )
        self.REQUEST_PREDICTION_ST24hours = Gauge(
            'request_prediction_st24hours', 'Prediction Result',
            ['app_name', 'method', 'endpoint']
        )
        self.REQUEST_PREDICTION_ST168hours = Gauge(
            'request_prediction_st168hours', 'Prediction Result',
            ['app_name', 'method', 'endpoint']
        )

        #Model Version Info
        self.MODEL_VERSION_CT3hours = Gauge(
            'model_version_ct3hours', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_VERSION_CT24hours = Gauge(
            'model_version_ct24hours', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_VERSION_CT168hours = Gauge(
            'model_version_ct168hours', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )

        self.MODEL_VERSION_ST3hours = Gauge(
            'model_version_st3hours', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_VERSION_ST24hours = Gauge(
            'model_version_st24hours', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_VERSION_ST168hours = Gauge(
            'model_version_st168hours', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )

        #Model Retrain Total Info
        self.MODEL_RETRAIN_CT3hours = Counter(
            'model_retrain_ct3hours', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_RETRAIN_CT24hours = Counter(
            'model_retrain_ct24hours', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_RETRAIN_CT168hours = Counter(
            'model_retrain_ct168hours', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )

        self.MODEL_RETRAIN_ST3hours = Counter(
            'model_retrain_st3hours', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_RETRAIN_ST24hours = Counter(
            'model_retrain_st24hours', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_RETRAIN_ST168hours = Counter(
            'model_retrain_st168hours', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )

        #Model Retraining Status
        self.MODEL_STATUS_CT3hours = Enum(
            'model_status_ct3hours', 'Model Status',
            states=['idle', 'retraining']
        )
        self.MODEL_STATUS_CT24hours = Enum(
            'model_status_ct24hours', 'Model Status',
            states=['idle', 'retraining']
        )
        self.MODEL_STATUS_CT168hours = Enum(
            'model_status_ct168hours', 'Model Status',
            states=['idle', 'retraining']
        )

        self.MODEL_STATUS_ST3hours = Enum(
            'model_status_st3hours', 'Model Status',
            states=['idle', 'retraining']
        )
        self.MODEL_STATUS_ST24hours = Enum(
            'model_status_st24hours', 'Model Status',
            states=['idle', 'retraining']
        )
        self.MODEL_STATUS_ST168hours = Enum(
            'model_status_st168hours', 'Model Status',
            states=['idle', 'retraining']
        )
        