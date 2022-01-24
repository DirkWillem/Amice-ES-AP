from typing import Optional
import math
import dataclasses

import numpy as np

from . import feature_template as ft, feature, timeline


@dataclasses.dataclass
class ProfileMatchOptions:
    """
    Contains the various options for matching a profile
    """
    tol_t: float = 5    # Time offset tolerance
    tol_f: float = 40   # Feature mismatch tolerance


class ApplianceProfile:
    """
    Represents the features of the power profile of a single appliance
    """
    tl: timeline.Timeline

    def __init__(self, tl:  timeline.Timeline):
        self.tl = tl

    def try_match_timeline(
            self,
            tl: timeline.Timeline,
            t0: float,
            options: Optional[ProfileMatchOptions] = None) -> (float, float, set[int]):
        """
        Attempts to match the profile to the timeline, with the first feature at t0
        :param tl: Timeline to match to
        :param t0: Start time to match the appliance profile to
        :param options: Match options
        :return: (Power error, time error, IDs of the matched feature)
        """

        # Default for options
        if options is None:
            options = ProfileMatchOptions()

        err_t = 0
        err_f = 0

        fids = set()

        # iterate over features of the appliance timeline
        for _, tf, feat in self.tl.features:
            # Find all features in the aggregate timeline that could match the current feature within the time tolerance
            matches = tl.find_features(t0+tf-options.tol_t, t0+tf+options.tol_t)
            if len(matches) == 0:
                return math.inf, math.inf, {}

            # Iterate over all found features and find the one with the best match
            best_id = -1
            best_err_f = math.inf
            best_err_t = math.inf
            for mid, mt, mf in matches:
                err_f = mf.compare(feat)
                if err_f > options.tol_f:
                    continue
                if err_f < best_err_f:
                    best_id, best_err_f, best_err_t = mid, err_f, abs(t0+tf-mt)

            # This specific feature was not matched, so the entire profile cannot be matched
            if best_id == -1:
                return math.inf, math.inf, {}

            # Add the best match to the set of matched features
            err_f += best_err_f
            err_t += best_err_t
            fids.add(best_id)

        return err_f, err_t, fids


def create_timeline(t, y, window: int = 5) -> timeline.Timeline:
    """
    Creates a timeline from the given time and power data
    :param t: Time data
    :param y: Power data
    :param window: Window size in which to match features
    :return: Timeline instance
    """

    tmpls = [
        ft.LoadStepTemplate(step_tol=50)
    ]

    tl = timeline.Timeline()

    i = 0
    while i < len(y) - window:
        for tmpl in tmpls:
            is_match, tm, feat = tmpl.match_feature(t[i:i + window], y[i:i + window])
            if is_match:
                tl.add_feature(tm, feat)
                i += window
            else:
                i += 1

    return tl


def create_appliance_profile(fname: str, window=5) -> ApplianceProfile:
    """
    Creates an appliance timeline from a CSV file
    :param fname: File name of the appliance profile
    :param window: Window size in which to match features
    :return: Appliance profile
    """

    data = np.genfromtxt(fname)
    t = np.arange(len(data))

    tl = create_timeline(t, data, window)

    return ApplianceProfile(tl)
