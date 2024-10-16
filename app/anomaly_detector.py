# anomaly_detector.py
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

class AnomalyDetector:
    """
    A hybrid anomaly detection system combining statistical and machine learning approaches.
    
    This class implements two methods of anomaly detection:
    1. Rolling Z-Score: For real-time detection of point anomalies
    2. Isolation Forest: For detecting complex anomaly patterns in batches
    
    Attributes:
        window_size (int): Size of the rolling window for statistics
        alpha (float): Smoothing factor for exponential moving average (0 < alpha < 1)
        z_threshold (float): Number of standard deviations for Z-score anomaly threshold
        n_freq (int): Number of frequencies to retain in seasonal decomposition
        if_batch_size (int): Batch size for Isolation Forest processing
        if_estimators (int): Number of trees in Isolation Forest ensemble
    """
    
    def __init__(self, 
                 window_size: int = 50, 
                 alpha: float = 0.1, 
                 z_threshold: float = 3, 
                 n_freq: int = 5,
                 if_batch_size: int = 200, 
                 if_estimators: int = 100) -> None:
        """Initialize the anomaly detector with configuration parameters."""
        # Validate input parameters
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        if window_size < 1:
            raise ValueError("Window size must be positive")
        
        self.window_size = window_size
        self.alpha = alpha
        self.z_threshold = z_threshold
        self.n_freq = n_freq
        self.if_batch_size = if_batch_size
        self.if_estimators = if_estimators
        self.isolation_forest = IsolationForest(
            n_estimators=if_estimators,
            contamination='auto',
            random_state=42
        )
        logger.info("AnomalyDetector initialized with window_size=%d, alpha=%.2f", 
                   window_size, alpha)
        
    def ema(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            values: Array of numerical values
            
        Returns:
            Array of EMA values same length as input
        """
        if len(values) == 0:
            raise ValueError("Input array must not be empty")
            
        ema_values = np.zeros_like(values)
        ema_values[0] = values[0]
        for i in range(1, len(values)):
            ema_values[i] = self.alpha * values[i] + (1 - self.alpha) * ema_values[i-1]
        return ema_values
    
    def mad(self, data: npt.NDArray[np.float64]) -> float:
        """
        Calculate Median Absolute Deviation.
        
        Args:
            data: Input array
            
        Returns:
            MAD value for the input array
        """
        if len(data) < self.window_size:
            raise ValueError(f"Data length must be >= window_size ({self.window_size})")
            
        recent_data = data[-self.window_size:]
        median = np.median(recent_data)
        return np.median(np.abs(recent_data - median))
    
    def remove_seasonality(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Remove seasonality using Fourier Transform.
        
        Args:
            data: Input time series data
            
        Returns:
            Deseasonalized data array
        """
        if len(data) < 2 * self.n_freq:
            raise ValueError(f"Data length must be >= 2 * n_freq ({2 * self.n_freq})")
            
        data_fft = fft(data)
        data_fft[self.n_freq:-self.n_freq] = 0  # Keep only n_freq lowest frequencies
        seasonal_component = np.real(ifft(data_fft))
        return data - seasonal_component
    
    def rolling_z_score(self, data: npt.NDArray[np.float64]) -> List[int]:
        """
        Detect anomalies using Rolling Z-Score method.
        
        Args:
            data: Input time series data
            
        Returns:
            List of indices where anomalies were detected
        """
        if len(data) < self.window_size:
            logger.warning("Data length less than window size, returning empty anomaly list")
            return []
            
        anomalies = []
        for i in range(self.window_size, len(data)):
            window = data[i-self.window_size:i]
            mean = np.mean(window)
            std = np.std(window)
            
            if std == 0:  # Avoid division by zero
                continue
                
            z_score = (data[i] - mean) / std
            if np.abs(z_score) > self.z_threshold:
                anomalies.append(i)
                logger.debug(f"Z-score anomaly detected at index {i}: {z_score:.2f}")
        
        return anomalies

    def isolation_forest_detect(self, data: npt.NDArray[np.float64]) -> List[int]:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            data: Input time series data
            
        Returns:
            List of indices where anomalies were detected
        """
        try:
            # Reshape data for sklearn
            reshaped_data = data.reshape(-1, 1)
            
            # Process in batches to handle large datasets
            anomalies = []
            for i in range(0, len(data), self.if_batch_size):
                batch = reshaped_data[i:i + self.if_batch_size]
                predictions = self.isolation_forest.fit_predict(batch)
                batch_anomalies = np.where(predictions == -1)[0] + i
                anomalies.extend(batch_anomalies.tolist())
            
            logger.info(f"Isolation Forest detected {len(anomalies)} anomalies")
            return sorted(anomalies)
            
        except Exception as e:
            logger.error(f"Error in Isolation Forest detection: {str(e)}")
            return []

    def detect(self, data_stream: npt.NDArray[np.float64]) -> Tuple[List[int], Optional[npt.NDArray[np.float64]]]:
        """
        Main method to detect anomalies using both detection methods.
        
        Args:
            data_stream: Input time series data
            
        Returns:
            Tuple containing:
                - List of anomaly indices
                - Processed data array (or None if error occurs)
        """
        try:
            logger.info(f"Processing data stream of length {len(data_stream)}")
            
            # Preprocess data
            detrended_data = self.remove_seasonality(data_stream)
            ema_data = self.ema(detrended_data)
            
            # Detect anomalies using both methods
            z_score_anomalies = self.rolling_z_score(ema_data)
            if_anomalies = self.isolation_forest_detect(ema_data)
            
            # Combine and sort anomalies
            all_anomalies = sorted(set(z_score_anomalies) | set(if_anomalies))
            
            logger.info(f"Total anomalies detected: {len(all_anomalies)}")
            return all_anomalies, ema_data
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return [], None