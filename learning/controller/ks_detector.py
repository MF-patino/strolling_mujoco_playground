from collections import deque
from scipy.stats import ks_2samp
from river.drift import ADWIN
import numpy as np

class KSDriftDetector:
    def __init__(self, total_size=1000, window_size=250, adwin_delta=1e-3, 
                 performance_alert_threshold=1e-4, hertz=50):
        """
        KS-based drift detector with ADWIN on the KS statistic.

        Args:
            total_size (int): Total buffer length (Reference + Window).
            window_size (int): Size of the current window.
            adwin_delta (float): ADWIN confidence parameter.
                                 Smaller = fewer false alarms.
        """
        self.hertz = hertz
        self.performance_alert_threshold = performance_alert_threshold
        self.buffer = deque(maxlen=total_size)
        self.stat_values = []
        self.window_size = window_size
        
        # Minimum samples needed before we start testing
        # We need at least full window + some reference
        self.min_samples = window_size * 2

        # ADWIN monitors the statistic stream
        self.adwin = ADWIN(delta=adwin_delta)

    def checkPolicyPerformance(self, data, expected_error_info):
        '''
        Checks for really salient instabilities that could lead to the robot falling
        over in the near future.
        '''
        # We will only look at data from the current second
        # This focuses the statistical detection on current perturbations
        # being experienced by the robot
        last_errors = data[-self.hertz:]
        mean_native_error, native_errors = expected_error_info

        # If the slightest negative perturbation is detected, check if
        # error distributions are different
        if mean_native_error < np.mean(last_errors):
            # Here the KS test requires a high amount of evidence/certainty
            # to declare that the policy is out of its environment AND experiencing
            # instabilities that could lead to the robot falling over
            _, p_value = ks_2samp(native_errors, last_errors)
            if p_value < self.performance_alert_threshold:
                return True
        
        return False
        
    def update(self, error_val, expected_error_info):
        """
        Input: error_val (float)
        Returns: is_drift (bool), statistic (float)
        """
        self.buffer.append(error_val)
        data = list(self.buffer)

        # Assess policy performance against the baseline performance in its native environment
        policy_performance_alert = self.checkPolicyPerformance(data, expected_error_info)

        # If we don't have enough data for drift detection yet and a performance alert is fired,
        # then this was not a valid domain drift detection. 
        # We handle this case in the controller logic as a need to adapt a new policy, since the
        # policy chosen by the GP, which should be the best we have in the catalog, has dangerous instabilities.
        if len(data) < self.min_samples:
            self.stat_values.append(0)
            return False, 0., policy_performance_alert
        elif policy_performance_alert:
            self.reset(data)
            return True, 0., policy_performance_alert
        
        # Reference: Everything EXCEPT the last N elements
        # Window: The last N elements
        reference_data = data[:-self.window_size]
        window_data = data[-self.window_size:]

        # Run KS Test
        statistic, _ = ks_2samp(reference_data, window_data)
        # Dynamic smoothing of statistic for plotting and ADWIN detector
        alpha = 0.1
        statistic = alpha * statistic + (1-alpha) * self.stat_values[-1]

        self.stat_values.append(statistic)
        
        self.adwin.update(statistic)
        is_drift = False
        # If ADWIN detects drift and the errors are worse recently
        if self.adwin.drift_detected and np.mean(reference_data) < np.mean(window_data):
            # And if the distributions are becoming more different recently
            if np.mean(self.stat_values[-len(data):-self.window_size]) < np.mean(self.stat_values[-self.window_size:]):
                is_drift = True

        if is_drift:
            self.reset(data)

        return is_drift, statistic, False
    
    # The reference data at the point of a domain change detection is filled with the previous domain's prediction errors. 
    # This is stale data as now we are only concerned about the data from the new domain the robot is in.
    # In this method, the reference data is cleared and the adwin detector is also reset
    def reset(self, data):
        self.buffer.clear()
        self.adwin._reset()