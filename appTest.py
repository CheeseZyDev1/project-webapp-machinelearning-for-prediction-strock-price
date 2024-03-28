from flask import Flask, render_template, jsonify, abort
import yfinance as yf
from flask_apscheduler import APScheduler
from tensorflow.keras.models import load_model
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

scheduler = APScheduler()
scheduler.init_app(app)

model_path = r'C:\Users\plam\Desktop\project ml\modelForML.h5'
model = load_model(model_path)

stocks_data = {
    'AOT.BK': {'current_price': None, 'predicted_price': None},
    'KBANK.BK': {'current_price': None, 'predicted_price': None},
    'XPG.BK': {'current_price': None, 'predicted_price': None},
    'KTB.BK': {'current_price': None, 'predicted_price': None},
    'PTTEP.BK': {'current_price': None, 'predicted_price': None},
    # ... other stocks
}

def prepare_data_for_prediction(ticker):
    data = yf.download(ticker, period="1d", interval="1m")
    return data['Close'].iloc[-1]  # Use the last close price

def update_and_predict():
    with app.app_context():
        for ticker in stocks_data.keys():
            try:
                current_price = prepare_data_for_prediction(ticker)
                predicted_price = model.predict(np.array([[current_price]]))[0]
                stocks_data[ticker] = {
                    'current_price': current_price.item(),  # Convert numpy float to Python float
                    'predicted_price': predicted_price.item()  # Convert numpy float to Python float
                }
            except Exception as e:
                logging.error(f"Error updating stock data for {ticker}: {e}")

@app.route('/get_stock_data/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    stock_info = stocks_data.get(ticker)
    if stock_info:
        return jsonify(stock_info)
    else:
        abort(404, description="Stock ticker not found.")

@app.route('/')
def index():
    return render_template('index.html', stocks=stocks_data.keys())

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404

if __name__ == '__main__':
    scheduler.add_job(func=update_and_predict, trigger='interval', minutes=5, id='stock_pred_job')
    scheduler.start()
    app.run(debug=True)
