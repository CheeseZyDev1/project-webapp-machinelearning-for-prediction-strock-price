from flask import Flask, render_template, request, jsonify, abort
import yfinance as yf
from flask_apscheduler import APScheduler
from tensorflow.keras.models import load_model
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize the APScheduler
scheduler = APScheduler()
scheduler.init_app(app)

# Assuming Orthogonal is an initializer you might need
# from tensorflow.keras.initializers import Orthogonal
# custom_objects = {'Orthogonal': Orthogonal(gain=1.0, seed=None)}

# Define your model here
model_path = r'C:\Users\plam\Desktop\project ml\modelForML.h5'
model = load_model(model_path)  # If you need custom objects, add them here

# Placeholder for your stock data
stocks_data = {
    'AOT.BK': {'current_price': None, 'predicted_price': None},
    'KBANK.BK': {'current_price': None, 'predicted_price': None},
    'KTB.BK': {'current_price': None, 'predicted_price': None},
    'PTTEP.BK': {'current_price': None, 'predicted_price': None},
    'XPG.BK': {'current_price': None, 'predicted_price': None},
    # ... other stocks
}

def prepare_data_for_prediction(ticker):
    data = yf.download(ticker, period="10m", interval="1m")
    processed_data = data[-1:].values.reshape(1, -1)
    return processed_data

def update_and_predict():
    with app.app_context():  # Required for background tasks that use `jsonify`
        for ticker in stocks_data.keys():
            try:
                current_data = prepare_data_for_prediction(ticker)
                predicted_price = model.predict(current_data).item()  # Assuming your model returns a single value
                
                stocks_data[ticker]['current_price'] = float(current_data[-1, -1])
                stocks_data[ticker]['predicted_price'] = predicted_price
            except Exception as e:
                logging.error(f"Error updating stock data for {ticker}: {e}")

@app.route('/get_stock_data/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    stock_info = stocks_data.get(ticker)
    if stock_info:
        # Serialize NumPy objects to native Python types
        return jsonify({
            'current_price': float(stock_info['current_price']) if stock_info['current_price'] else None,
            'predicted_price': float(stock_info['predicted_price']) if stock_info['predicted_price'] else None
        })
    else:
        abort(404, description="Stock ticker not found.")

@app.route('/')
def index():
    return render_template('index.html', stocks=list(stocks_data.keys()))

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404

if __name__ == '__main__':
    scheduler.add_job(func=update_and_predict, trigger='interval', minutes=5, id='stock_pred_job')
    scheduler.start()
    app.run(debug=True)
