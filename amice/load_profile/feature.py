import abc
import math


class Feature(abc.ABC):
    """
    Abstract Base Class for all features
    """

    @abc.abstractmethod
    def compare(self, other: 'Feature') -> float:
        pass


class LoadStepFeature(Feature):
    """
    Represents a step in power load
    """

    dp: float

    def compare(self, other: 'Feature') -> float:
        """
        Compares this feature to another feature instance and returns the error between the two
        :param other: Other feature to compare to
        :return: Comparison error
        """
        if isinstance(other, LoadStepFeature):
            return abs(self.dp - other.dp)

        return math.inf

    def __init__(self, dp: float):
        self.dp = dp

    def __str__(self):
        return f"Step (dp={self.dp} W)"
