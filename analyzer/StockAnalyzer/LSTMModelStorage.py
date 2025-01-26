import os
import joblib
from tensorflow.keras.models import load_model
import json


class ModelStorage:
    def __init__(self, base_path='models'):

        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save_model(self, model, volume_scaler, price_scaler, binary_encoder, model_name='stock_prediction',
                   additional_params=None):

        model_dir = os.path.join(self.base_path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'model.keras')
        model.save(model_path)

        scaler_path = os.path.join(model_dir, 'price_scaler.pkl')
        joblib.dump(price_scaler, scaler_path)

        scaler_path = os.path.join(model_dir, 'volume_scaler.pkl')
        joblib.dump(volume_scaler, scaler_path)

        encoder_path = os.path.join(model_dir, 'encoder.pkl')
        joblib.dump(binary_encoder, encoder_path)

        if additional_params:
            params_path = os.path.join(model_dir, 'params.json')
            with open(params_path, 'w') as f:
                json.dump(additional_params, f)

    def load_model(self, model_name='stock_prediction'):

        model_dir = os.path.join(self.base_path, model_name)

        model_path = os.path.join(model_dir, 'model.keras')
        model = load_model(model_path)

        price_scaler_path = os.path.join(model_dir, 'price_scaler.pkl')
        price_scaler = joblib.load(price_scaler_path)

        volume_scaler_path = os.path.join(model_dir, 'volume_scaler.pkl')
        volume_scaler = joblib.load(volume_scaler_path)

        encoder_path = os.path.join(model_dir, 'encoder.pkl')
        binary_encoder = joblib.load(encoder_path)

        params_path = os.path.join(model_dir, 'params.json')
        additional_params = None
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                additional_params = json.load(f)

        return price_scaler, volume_scaler, binary_encoder, model, additional_params

    def list_saved_models(self):

        if not os.path.exists(self.base_path):
            return []
        return [d for d in os.listdir(self.base_path)
                if os.path.isdir(os.path.join(self.base_path, d))]
