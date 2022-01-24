import os
import numpy as np
import matplotlib.pyplot as plt
import amice.load_profile as lp
import math

import scipy.io as sio

if __name__ == '__main__':
    # Create appliance profiles from individual datasets
    data_dir = 'data/appliances'

    profiles = dict()
    for fname in os.listdir(data_dir):
        if fname.endswith('.csv'):
            name = os.path.splitext(fname)[0]
            profiles[name] = lp.create_appliance_profile(os.path.join(data_dir, fname))

    # Load aggregate load profile
    test_profile = sio.loadmat('data/data5.mat')
    test_t = test_profile['X'].tolist()[0]
    test_p = test_profile['Y'].tolist()[0]

    tl = lp.create_timeline(test_t, test_p)

    # Print the contents of the aggregate profile
    test_r = test_profile['r'].tolist()[0]
    test_t = test_profile['xx'].tolist()[0]

    names = ['', 'dishwasher', 'PC', 'Oven 180', 'Refrigerator', 'Toaster', 'TV',
             'Washing Machine 30C', 'Washing Machine 40C', 'Water Heater']
    print('DATA CONTAINS:')
    for r, t0 in zip(test_r, test_t):
        print(f'  - {names[r]} at t={t0}')
    print('')

    # Iterate over all features & appliances until either all features are matched or no profile can be matched
    #  this is very inefficient, but it works (kinda)
    print('FOUND DATA:')
    while len(tl.features) > 0:
        # Keep track of the best match so far
        min_err = math.inf
        min_feat = None
        min_t = 0
        min_ids = {}
        min_name = ''

        # Iterate over features
        for i in range(len(tl.features)):
            fid, t, f = tl.features[i]

            # Iterate over profiles
            for pn, p in profiles.items():
                # Attempt to match the profile to the sequence of features starting at feature i
                ef, et, ids = p.try_match_timeline(tl, t)
                e = ef+et

                # If the error of the matched feature is better than the best one yet, store it as the new best one
                if e < min_err:
                    min_err = e
                    min_feat = p
                    min_t = t
                    min_ids = ids
                    min_name = pn

        if not math.isinf(min_err):
            # If we have a non-zero best error, print the matched profile
            print(f'  - Found profile "{min_name}" at t={min_t+tl.t0-min_feat.tl.t0} (err={min_err}, {len(min_ids)} features)')

            # Remove the matched features from the aggregate timeline
            tl.remove_features(min_ids)
        else:
            # Infinite error, so no feature matched. Terminate.
            print(f"  * no more matches found, {len(tl.features)} unmatched features. Terminating...")
            break