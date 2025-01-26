from DataStorage import DataStorage
import pandas as pd
from ta.momentum import (
    RSIIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
    PercentagePriceOscillator,
    ROCIndicator
)
from ta.trend import (
    MACD,
    CCIIndicator,
    SMAIndicator,
    EMAIndicator,
    WMAIndicator,
    TRIXIndicator
)


class TechnicalAnalyzer:
    def __init__(self):
        self.storage = DataStorage()
        self.thresholds = {
            'RSI': {'buy': 25, 'sell': 75},
            'Stoch_%K': {'buy': 15, 'sell': 85},
            'Williams_R': {'buy': -85, 'sell': -15},
            'PPO': {'buy': -1.5, 'sell': 1.5},
            'ROC': {'buy': -3, 'sell': 3},
            'CCI': {'buy': -150, 'sell': 150}
        }

    def generate_oscillator_signal(self, row, indicator_name):
        value = row[indicator_name]

        if pd.isna(value):
            return 'Hold'

        thresholds = self.thresholds[indicator_name]

        if value <= thresholds['buy']:
            return 'Buy'
        elif value >= thresholds['sell']:
            return 'Sell'
        return 'Hold'

    def generate_moving_average_signal(self, data, price_col, ma_col):
        signals = pd.Series('Hold', index=data.index)

        buffer = data[ma_col] * 0.01

        signals[data[price_col] > (data[ma_col] + buffer)] = 'Buy'
        signals[data[price_col] < (data[ma_col] - buffer)] = 'Sell'

        return signals

    def generate_macd_signal(self, row):
        macd = row['MACD']
        macd_signal = row['MACD_Signal']

        if pd.isna(macd) or pd.isna(macd_signal):
            return 'Hold'

        threshold = abs(macd_signal) * 0.15

        if macd > (macd_signal + threshold):
            return 'Buy'
        elif macd < (macd_signal - threshold):
            return 'Sell'
        return 'Hold'

    def preprocess_data(self, data):
        numeric_cols = ['Close', 'High', 'Low', 'Avg. Price', '%chg.',
                        'Turnover in BEST in denars', 'Total turnover in denars', 'Volume']

        for col in numeric_cols:
            data[col] = (
                data[col]
                .str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
                .astype(float)
            )

        data['Volume'] = data['Volume'].astype(int)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date', ascending=True)
        data.set_index('Date', inplace=True)
        return data

    def compute_indicators(self, data):
        # Oscillators
        data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
        stoch_indicator = StochasticOscillator(
            high=data['High'], low=data['Low'], close=data['Close'], window=14
        )
        data['Stoch_%K'] = stoch_indicator.stoch()
        data['Stoch_%D'] = stoch_indicator.stoch_signal()

        data['Williams_R'] = WilliamsRIndicator(
            high=data['High'], low=data['Low'], close=data['Close'], lbp=14
        ).williams_r()

        ppo_indicator = PercentagePriceOscillator(close=data['Close'])
        data['PPO'] = ppo_indicator.ppo()
        data['PPO_Signal'] = ppo_indicator.ppo_signal()

        data['ROC'] = ROCIndicator(close=data['Close'], window=12).roc()

        data['CCI'] = CCIIndicator(
            high=data['High'], low=data['Low'], close=data['Close'], window=20
        ).cci()

        # Moving averages
        data['SMA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
        data['EMA_20'] = EMAIndicator(close=data['Close'], window=20).ema_indicator()
        data['WMA_20'] = WMAIndicator(close=data['Close'], window=20).wma()
        data['MACD'] = MACD(close=data['Close']).macd()
        data['MACD_Signal'] = MACD(close=data['Close']).macd_signal()
        data['TRIX'] = TRIXIndicator(close=data['Close'], window=20).trix()
        return data

    def analyze_stock(self, issuer):
        db = self.storage.get_by_issuer(issuer)
        columns = [
            'Date', 'Issuer', 'Avg. Price', 'Close', 'High', 'Low', '%chg.',
            'Total turnover in denars', 'Turnover in BEST in denars', 'Volume'
        ]
        data = pd.DataFrame(db, columns=columns)

        data = self.preprocess_data(data)
        data = self.compute_indicators(data)

        oscillator_indicators = ['RSI', 'Stoch_%K', 'Williams_R', 'PPO', 'ROC', 'CCI']
        for indicator in oscillator_indicators:
            data[f'{indicator}_Signal'] = data.apply(
                self.generate_oscillator_signal, indicator_name=indicator, axis=1
            )

        ma_indicators = [('Close', 'SMA_20'), ('Close', 'EMA_20'),
                         ('Close', 'WMA_20'), ('Close', 'TRIX')]
        for price_col, ma_col in ma_indicators:
            data[f'{ma_col}_Signal'] = self.generate_moving_average_signal(data, price_col, ma_col)

        data['MACD_Signal'] = data.apply(self.generate_macd_signal, axis=1)

        data_weekly = data.resample('W').last()
        data_monthly = data.resample('ME').last()

        latest_daily_signal = data.iloc[-1].filter(like='_Signal')
        latest_weekly_signal = data_weekly.iloc[-1].filter(like='_Signal')
        latest_monthly_signal = data_monthly.iloc[-1].filter(like='_Signal')

        return {
            'daily': latest_daily_signal.to_dict(),
            'weekly': latest_weekly_signal.to_dict(),
            'monthly': latest_monthly_signal.to_dict()
        }
