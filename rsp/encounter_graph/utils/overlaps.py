"""Inverval intersection utils from
https://stackoverflow.com/questions/2953967/built-in-function-for-computing-
overlap-in-python."""
import numpy as np


def overlap_interval(interval1, interval2):
    """Given [0, 4] and [1, 10] returns [1, 4]"""

    if interval2[0] <= interval1[0] <= interval2[1]:
        start = interval1[0]
    elif interval1[0] <= interval2[0] <= interval1[1]:
        start = interval2[0]
    else:
        return None

    if interval2[0] <= interval1[1] <= interval2[1]:
        end = interval1[1]
    elif interval1[0] <= interval2[1] <= interval1[1]:
        end = interval2[1]
    else:
        return None
    # corner case of infinity bounds
    if start > -np.inf and end < np.inf:
        return (start, end)
    return None


def overlaps(a, b):
    """Return the amount of overlap, in bp between a and b.

    If >0, the number of bp of overlap If 0,  they are book-ended. If
    <0, the distance in bp between them
    """

    return min(a[1], b[1]) - max(a[0], b[0])
