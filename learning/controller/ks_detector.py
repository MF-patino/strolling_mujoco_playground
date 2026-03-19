from collections import deque
from scipy.stats import ks_2samp
from river.drift import ADWIN
import numpy as np

class KSDriftDetector:
    def __init__(self, total_size=1000, window_size=250, adwin_delta=1e-3):
        """
        KS-based drift detector with ADWIN on the KS statistic.

        Args:
            total_size (int): Total buffer length (Reference + Window).
            window_size (int): Size of the current window.
            adwin_delta (float): ADWIN confidence parameter.
                                 Smaller = fewer false alarms.
        """
        self.buffer = deque(maxlen=total_size)
        self.stat_values = []
        self.window_size = window_size
        
        # Minimum samples needed before we start testing
        # We need at least full window + some reference
        self.min_samples = window_size * 2

        # ADWIN monitors the statistic stream
        self.adwin = ADWIN(delta=adwin_delta)

    def update(self, error_val, expected_error_info):
        """
        Input: error_val (float)
        Returns: is_drift (bool), statistic (float)
        """
        self.buffer.append(error_val)
        data = list(self.buffer)

        # If we don't have enough data for drift detection yet, assess policy performance
        # against the baseline performance in its native environment
        if len(self.buffer) < self.min_samples:
            mean_native_error, native_errors = expected_error_info
            last_errors = data[-50:]
            _, p_value = ks_2samp(native_errors, last_errors)
            policy_performance_alert = p_value < 1e-4 and mean_native_error < np.mean(last_errors)
            
            self.stat_values.append(0)
            return False, 0., policy_performance_alert
        
        # Reference: Everything EXCEPT the last N elements
        # Window: The last N elements
        reference_data = data[:-self.window_size]
        window_data = data[-self.window_size:]

        # Run KS Test
        statistic, _ = ks_2samp(reference_data, window_data)
        alpha = 0.1
        statistic = alpha * statistic + (1-alpha) * self.stat_values[-1]

        self.stat_values.append(statistic)
        
        self.adwin.update(statistic)
        is_drift = self.adwin.drift_detected

        if is_drift:
            self.reset(data)

        return is_drift, statistic, False
    
    # The reference data at the point of a domain change detection is filled with the previous domain's prediction errors. 
    # This is stale data as now we are only concerned about the data from the new domain the robot is in.
    # In this method, the reference data is cleared and the adwin detector is also reset
    def reset(self, data):
        self.buffer.clear()
        #self.buffer.extend(data[-self.window_size:])
        self.adwin._reset()