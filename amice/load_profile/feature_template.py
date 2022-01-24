import abc
import numpy as np
from . import feature


class FeatureTemplate(abc.ABC):
    @abc.abstractmethod
    def match_feature(self, t: np.array, y: np.array) -> (bool, float, dict[str, float]):
        pass


class LoadStepTemplate(FeatureTemplate):
    """
    Template of a load step feature
    """
    def __init__(self, step_tol: float):
        self.step_tol = step_tol

    def match_feature(self, t: np.array, y: np.array) -> (bool, float, feature.Feature):
        """
        Matches a load step feature from the given time and power data
        :param t: Time data
        :param y: Power data
        :return: (match, time, feat) where:
                    - match is True when a feature is matched, False otherwise
                    - time contains the exact time at which the feature was matched within the data
                    - feature contains a Feature instance representing the matched feature
        """
        delta = abs(y[0] - y[-1])

        # Step over window smaller than tolerance, no match
        if delta < self.step_tol:
            return False, 0, dict()

        # Trim start of the signal
        t0 = 0
        for i in range(1, len(t)):
            t0 = i
            if abs(y[i - 1] - y[i]) > self.step_tol:
                break

        # Trim end of signal
        t1 = len(t)-1
        for i in range(len(t)-1, 1, -1):
            t1 = i
            if abs(y[i - 1] - y[i]) > self.step_tol:
                break

        if delta > self.step_tol:
            return True, (t[t0] + t[t1])/2, feature.LoadStepFeature(y[-1] - y[0])

        return False, 0, dict()
