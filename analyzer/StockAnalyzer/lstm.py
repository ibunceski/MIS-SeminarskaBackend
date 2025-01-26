from DataStorage import DataStorage
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
from LSTMModelStorage import ModelStorage
import category_encoders as ce


class LSTMAnalyzer:
    def __init__(self):
        self.n_lags = 20
        self.price_scaler = StandardScaler()
        self.volume_scaler = StandardScaler()
        self.binary_encoder = ce.BinaryEncoder()
        self.model = None
        self.feature_dims = None

        self.model_storage = ModelStorage()
        self.data_storage = DataStorage()

        self.price_features = ['Close', 'High', 'Low', 'Avg Price', 'MA5', 'MA20']
        self.volume_features = ['Volume', 'Turnover in BEST in denars', 'Total turnover in denars']
        self.tech_features = ['RSI', 'MACD', 'Price_to_MA5', 'Price_to_MA20']

        self.columns = [
            'Date', 'Issuer', 'Avg Price', 'Close', 'High', 'Low', '%chg.',
            'Total turnover in denars', 'Turnover in BEST in denars', 'Volume'
        ]

        self.numeric_cols = ['Close', 'High', 'Low', 'Avg Price', '%chg.',
                             'Turnover in BEST in denars', 'Total turnover in denars', 'Volume']

        self.price_dims = len(self.price_features)
        self.volume_dims = len(self.volume_features)
        self.tech_dims = len(self.tech_features)
        self.issuer_dims = -1

    def load_model(self, price_scaler, volume_scaler, encoder, model, enc_len):
        self.price_scaler = price_scaler
        self.volume_scaler = volume_scaler
        self.binary_encoder = encoder
        self.model = model
        self.issuer_dims = enc_len

    def prepare_data(self, data):
        df = pd.DataFrame(data, columns=self.columns) if not isinstance(data, pd.DataFrame) else data.copy()

        for col in self.numeric_cols:
            df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Date', 'Issuer'], ascending=True)
        df['unique_index'] = df['Date'].astype(str) + '_' + df['Issuer']
        df.set_index('unique_index', inplace=True)

        for issuer in df['Issuer'].unique():
            mask = df['Issuer'] == issuer

            df.loc[mask, 'MA5'] = df.loc[mask, 'Close'].rolling(window=5).mean()
            df.loc[mask, 'MA20'] = df.loc[mask, 'Close'].rolling(window=20).mean()

            df.loc[mask, 'Price_to_MA5'] = df.loc[mask, 'Close'] / df.loc[mask, 'MA5'] - 1
            df.loc[mask, 'Price_to_MA20'] = df.loc[mask, 'Close'] / df.loc[mask, 'MA20'] - 1

            delta = df.loc[mask, 'Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=10).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=10).mean()
            rs = gain / loss
            df.loc[mask, 'RSI'] = 100 - (100 / (1 + rs))

            exp1 = df.loc[mask, 'Close'].ewm(span=12, adjust=False).mean()
            exp2 = df.loc[mask, 'Close'].ewm(span=26, adjust=False).mean()
            df.loc[mask, 'MACD'] = exp1 - exp2

        return df.dropna()

    def create_feature_matrix(self, data, training=True):
        for col in self.price_features[1:]:
            data[f'rel_{col}'] = data[col] / data['Close'] - 1

        if training:
            price_scaled = self.price_scaler.fit_transform(data[self.price_features])
            volume_scaled = self.volume_scaler.fit_transform(data[self.volume_features])
            encoded_issuer = self.binary_encoder.fit_transform(data[['Issuer']])
        else:
            price_scaled = self.price_scaler.transform(data[self.price_features])
            volume_scaled = self.volume_scaler.transform(data[self.volume_features])
            encoded_issuer = self.binary_encoder.transform(data[['Issuer']])

        feature_matrix = np.hstack([
            price_scaled,
            volume_scaled,
            data[self.tech_features].values,
            encoded_issuer.values
        ])

        if training:
            self.feature_dims = feature_matrix.shape[1]
            self.price_dims = len(self.price_features)
            self.volume_dims = len(self.volume_features)
            self.tech_dims = len(self.tech_features)
            self.issuer_dims = encoded_issuer.shape[1]

        return feature_matrix, price_scaled[:, 0]

    def prepare_sequences(self, data, training=True):
        feature_matrix, targets = self.create_feature_matrix(data, training)

        X, y = [], []
        for issuer in data['Issuer'].unique():
            issuer_mask = data['Issuer'] == issuer
            issuer_data = feature_matrix[issuer_mask]
            issuer_targets = targets[issuer_mask]

            if training:
                returns = np.diff(issuer_targets) / issuer_targets[:-1]
                for i in range(len(issuer_data) - self.n_lags - 1):
                    X.append(issuer_data[i:i + self.n_lags])
                    y.append(returns[i + self.n_lags])
            else:
                for i in range(len(issuer_data) - self.n_lags):
                    X.append(issuer_data[i:i + self.n_lags])
                    if i + self.n_lags < len(issuer_targets):
                        y.append(issuer_targets[i + self.n_lags])
                    else:
                        y.append(np.nan)

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, activation='tanh', return_sequences=True,
                 kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, activation='tanh',
                 kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='tanh')
        ])

        model.compile(optimizer='adam',
                      loss='huber',
                      metrics=['mae', 'mse'])
        return model

    def get_callbacks(self):
        return [
            EarlyStopping(monitor='val_loss',
                          patience=5,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=3,
                              min_lr=1e-6)
        ]

    def train(self, data, validation_split=0.2, epochs=30):
        df = self.prepare_data(data)

        df['target_return'] = df.groupby('Issuer')['Close'].pct_change()

        df = df[abs(df['target_return']) < 0.1]

        X, y = self.prepare_sequences(df, training=True)

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        time_weights = np.linspace(0.5, 1.0, len(X_train))

        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            sample_weight=time_weights,
            epochs=epochs,
            batch_size=32,
            callbacks=self.get_callbacks(),
            verbose=1
        )

        additional_params = {
            'n_lags': self.n_lags,
            'training_date': '2024-12-20',
            'model_version': '1.1',
            'enc_len': self.issuer_dims
        }

        self.model_storage.save_model(self.model, self.volume_scaler, self.price_scaler,
                                      self.binary_encoder, model_name='stock_model_good',
                                      additional_params=additional_params)

        return history

    def predict_next_days(self, data, days=5):
        df = self.prepare_data(data)
        X, _ = self.prepare_sequences(df, training=False)

        last_known_price = df['Close'].iloc[-1]
        last_sequence = X[-1]
        predictions = []
        running_price = last_known_price

        recent_prices = df['Close'].tail(20)
        recent_returns = recent_prices.pct_change().dropna()

        volatility = recent_returns.std()
        daily_volatility = volatility / np.sqrt(252)

        short_ma = recent_prices.tail(5).mean()
        long_ma = recent_prices.mean()
        trend_strength = (short_ma / long_ma - 1) * 100

        momentum = recent_returns.mean()

        running_momentum = momentum
        for day in range(days):
            predicted_return = self.model.predict(last_sequence[np.newaxis, :, :], verbose=0)[0][0]

            random_factor = np.random.normal(0, daily_volatility)

            trend_weight = 0.3
            momentum_weight = 0.2
            random_weight = 0.1

            time_decay = np.exp(-0.2 * day)

            blended_return = (
                    (1 - trend_weight - momentum_weight - random_weight) * predicted_return +
                    trend_weight * (trend_strength / 100) * time_decay +
                    momentum_weight * running_momentum * time_decay +
                    random_weight * random_factor
            )

            max_daily_move = min(0.03, 2 * daily_volatility)
            blended_return = np.clip(blended_return, -max_daily_move, max_daily_move)

            next_price = running_price * (1 + blended_return)

            if abs(next_price / short_ma - 1) > 0.05:
                reversion_strength = 0.3
                next_price = next_price * (1 - reversion_strength) + short_ma * reversion_strength

            predictions.append(next_price)

            price_scaled = self.price_scaler.transform([[next_price] + [0] * (self.price_dims - 1)])[0, 0]

            new_features = np.concatenate([
                [price_scaled] + [0] * (self.price_dims - 1),
                last_sequence[-1, self.price_dims:self.price_dims + self.volume_dims],
                last_sequence[-1,
                self.price_dims + self.volume_dims:self.price_dims + self.volume_dims + self.tech_dims],
                last_sequence[-1, -self.issuer_dims:]
            ])

            last_sequence = np.vstack([last_sequence[1:], new_features])
            running_price = next_price

            running_momentum = 0.7 * running_momentum + 0.3 * blended_return

        pct_change = (predictions[-1] - last_known_price) / last_known_price * 100

        if abs(pct_change) < 1.5:
            signal = "Neutral"
        else:
            confidence = min(abs(pct_change) / 3, 1)
            if pct_change > 0:
                signal = f"Positive (Confidence: {confidence:.2f})"
            else:
                signal = f"Negative (Confidence: {confidence:.2f})"

        return np.array(predictions), signal

    def plot_next_days(self, data, days=5):
        df = self.prepare_data(data)
        predictions, signal = self.predict_next_days(data, days)

        last_date = df['Date'].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days + 1)]

        all_dates = [last_date] + future_dates
        all_predictions = [df['Close'].iloc[-1]] + list(predictions)

        plt.figure(figsize=(12, 6))

        recent_data = df.tail(30)
        plt.plot(recent_data['Date'], recent_data['Close'], label='Actual', alpha=0.8)

        plt.plot(all_dates, all_predictions, label=f'Predicted ({signal})', linestyle='--', color='Red', alpha=0.8)

        last_price = df['Close'].iloc[-1]
        plt.axhline(y=last_price, color='r', linestyle=':', alpha=0.3)
        plt.text(recent_data['Date'].iloc[3], last_price, f'Last Price: {last_price:,.0f}',
                 verticalalignment='bottom', horizontalalignment='right')

        plt.title(f'Stock Price Prediction - {df["Issuer"].iloc[-1]}')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print(f"Issuer: {df['Issuer'].iloc[-1]}")
        print(f"Last Known Price: {last_price:,.0f}")
        print(f"Signal: {signal}")
        print("\nPredictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"Day {i}: {pred:,.0f} ({((pred / last_price) - 1) * 100:+.2f}%)")

    def data_for_plotting(self, data, days=5):
        df = self.prepare_data(data)
        predictions, signal = self.predict_next_days(data, days)

        last_date = df['Date'].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days + 1)]

        all_dates = [last_date] + future_dates
        all_predictions = [df['Close'].iloc[-1]] + list(predictions)

        recent_data = df.tail(30)

        last_price = df['Close'].iloc[-1]

        daily_percent = []

        for i, pred in enumerate(predictions, 1):
            daily_percent.append(f"Day {i}: {pred:,.0f} ({((pred / last_price) - 1) * 100:+.2f}%)")

        return {
            "dates": [str(date) for date in all_dates],
            "recentDates": [str(date) for date in recent_data['Date']],
            "actualPrices": recent_data['Close'].tolist(),
            "predictedPrices": all_predictions,
            "signal": signal,
            "issuer": df["Issuer"].iloc[-1],
            "lastPrice": last_price,
            "dailyPercent": daily_percent
        }

    def perform_prediction(self, issuer, days=5):
        model_params = self.model_storage.load_model("stock_model_good")
        self.load_model(model_params[0], model_params[1], model_params[2], model_params[3],
                        model_params[4]['enc_len'])
        data = self.data_storage.get_by_issuer(issuer)
        if len(data) < 100:
            print(f"Insufficient data for {issuer}")
            return

        return self.data_for_plotting(data, days)

    # def _perform_prediction_pltgraph(self, issuer, days=5):
    #     model_params = self.model_storage.load_model("stock_model_good")
    #     self.load_model(model_params[0], model_params[1], model_params[2], model_params[3],
    #                     model_params[4]['enc_len'])
    #     data = self.data_storage.get_by_issuer(issuer)
    #     if len(data) < 100:
    #         print(f"Insufficient data for {issuer}")
    #         return
    #
    #     self.plot_next_days(data, days)
