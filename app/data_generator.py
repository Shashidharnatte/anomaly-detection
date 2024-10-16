# data_generator.py
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

class DataGenerator:
    """
    Generates synthetic time series data for testing anomaly detection.
    
    The generated data includes:
    - Seasonal component (sine wave)
    - Linear trend
    - Random noise
    - Optional artificial anomalies
    """
    
    def __init__(self, seed: int = 42) -> None:
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        logger.info("DataGenerator initialized with seed %d", seed)
        
    def generate_stream(self, 
                       n_points: int = 1000, 
                       add_anomalies: bool = True) -> npt.NDArray[np.float64]:
        """
        Generate a simulated data stream.
        
        Args:
            n_points: Number of data points to generate
            add_anomalies: Whether to add artificial anomalies
            
        Returns:
            Array containing the generated time series
        """
        # Generate base components
        t = np.linspace(0, 10 * np.pi, n_points)
        seasonal = 10 * np.sin(t)
        trend = np.linspace(0, 5, n_points)
        noise = np.random.normal(0, 1, n_points)
        
        data = seasonal + trend + noise
        
        # Add artificial anomalies
        if add_anomalies:
            n_anomalies = n_points // 50  # Add anomaly every 50 points on average
            anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
            data[anomaly_indices] += np.random.normal(0, 10, n_anomalies)
            
        logger.info(f"Generated {n_points} points with {n_anomalies if add_anomalies else 0} artificial anomalies")
        return data