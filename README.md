Real-Time Anomaly Detection System

An anomaly detection system combining statistical and machine learning approaches to identify anomalies in time series data streams. This project demonstrates expertise in data processing, machine learning, and web development, with a focus on code quality and maintainability.

🌟 Features

•  Hybrid Detection Algorithm

-  Rolling Z-Score for real-time point anomaly detection
-  Isolation Forest for complex pattern detection
-  Seasonal adjustment using Fourier Transform
-  Exponential Moving Average smoothing


•  Interactive Dashboard

-  Real-time data visualization

-  Live statistical metrics

-  Responsive design

-  Automatic updates every 5 seconds


•  Production-Ready Architecture

-  Comprehensive error handling

-  Detailed logging system

-  Type hints and documentation

-  Modular design



🛠️ Technical Stack

• Backend

-  Python 3.8+
-  Flask 2.0.1
-  NumPy 1.21.0
-  SciPy 1.7.1
-  scikit-learn 1.0.1


• Frontend

-  HTML5/CSS3
-  JavaScript (ES6+)
-  Chart.js 3.7.0



📁 Project Structure

  anomaly-detection/
  ├── app/
  │   ├── __init__.py
  │   ├── anomaly_detector.py   # Core detection logic
  │   ├── data_generator.py     # Data simulation
  │   └── utils.py             # Utility functions
  ├── static/
  │   ├── css/
  │   │   └── style.css
  │   └── js/
  │       └── main.js
  ├── templates/
  │   └── index.html           # Dashboard template
  ├── tests/
  │   ├── __init__.py
  │   ├── test_detector.py
  │   └── test_generator.py
  ├── .gitignore
  ├── app.py                   # Flask application
  ├── config.py               # Configuration settings
  ├── requirements.txt        # Project dependencies
  └── README.md              # Project documentation


🚀 Quick Start

• Local Development

1. Clone the repository

git clone https://github.com/Shashidharnatte/anomaly-detection.git
cd anomaly-detection

2. Create a virtual environment

conda create -n env_name python==3.9.7 -y
conda activate env_name

3. Install dependencies

pip install -r requirements.txt


4. Run the application

python app.py

5. Open in browser

http://localhost:5000 or http://10.109.2.12:5000/
