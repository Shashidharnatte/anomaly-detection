import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.ensemble import IsolationForest
import logging
from typing import Tuple, List, Optional, Union
import numpy.typing as npt

# Configure logging with timestamp and log level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# app.py
from flask import Flask, render_template, jsonify
from app.data_generator import DataGenerator
from app.anomaly_detector import AnomalyDetector
import json
import os
from datetime import datetime

app = Flask(__name__)

# Initialize components
data_gen = DataGenerator()
detector = AnomalyDetector()

# Configure app
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')
app.config['DEBUG'] = os.environ.get('FLASK_ENV') == 'development'

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/data')
def get_data():
    """
    Generate and process new data stream.
    
    Returns:
        JSON containing:
        - Raw data points
        - Processed data points
        - Anomaly indices
        - Timestamp
    """
    try:
        # Generate new data
        data_stream = data_gen.generate_stream()
        
        # Detect anomalies
        anomalies, processed_data = detector.detect(data_stream)
        
        response = {
            'data': data_stream.tolist(),
            'processed_data': processed_data.tolist() if processed_data is not None else [],
            'anomalies': anomalies,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)