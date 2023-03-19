import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks


def find_levels(image, margin=40):
    # fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # ax[0].hist(image.ravel(), 251, [0,250])

    kde = stats.gaussian_kde(image.ravel()[image.ravel() <250])
    xx = np.linspace(0, 255, 100)
    yy = kde(xx)
    # ax[1].plot(xx, yy)

    valleys = find_peaks(-yy)[0]
    valleys = xx[valleys].astype(int)
    peaks = find_peaks(yy)[0]
    peaks = xx[peaks].astype(int)

    # print("V:", valleys, "P:", peaks)
    
    if len(valleys) == 0:
        # print("LVL 0")
        lvls = (60, 140, 200)

    elif len(valleys) == 1 or len(valleys) == 2:
        # print("LVL 1-2")
        valleys = np.insert(valleys, [0, len(valleys)], [0, 255])
        dist = np.vstack([peaks - valleys[:-1], valleys[1:] - peaks]).T
        mmin = np.minimum(dist.min(1, keepdims=True), 256//6)
        rem = dist - mmin

        ranges = np.insert(rem.flatten(), [0, len(valleys)], [0, 0])
        ranges = ranges[::2] + ranges[1::2]
        mask = np.repeat(ranges, 2)[1:-1].reshape(-1, 2)
        mask = (mask >= margin)

        new_dist = (dist - rem * mask).sum(1)
        extras = np.where(ranges >= margin)[0]
        lvls = np.insert(new_dist, extras, ranges[extras])
        lvls = np.cumsum(lvls)[:-1]
        # for i in lvls:
            # ax[1].axvline(x = i, color = 'r')
    else:
        # print("LVL 3")
        lvls = valleys

    # plt.show()
    return lvls