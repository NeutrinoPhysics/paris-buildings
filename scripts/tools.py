#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def distance(p1,p2):
    """
    orthodromic (geodesic) distance (in m) between point p1 and p2
    with 2-decimal precision (cm)
    """

    # --- convert lattitude and longitude
    #     from degrees to radians
    a = p1*np.pi/180.
    b = p2*np.pi/180.

    # --- take into account Earth's obloid deformation
    re = 6.378137e6     # earth radius in m at equator
    rp = 6.356752e6     # earth radius in m at pole
    flat = (re-rp)/re   # flattening parameter

    ra = re*(1.-flat*np.sin(a[1])**2) # Earth radius at p1's lattitude
    rb = re*(1.-flat*np.sin(b[1])**2) # Earth radius at p2's lattitude

    # --- compute distance using Haverside formula
    ortho = (ra+rb) * np.arcsin(np.sqrt(np.sin((b[1]-a[1])/2.)**2 + np.cos(a[1])*np.cos(b[1])*np.sin((b[0]-a[0])/2.)**2))
    ortho = np.round(ortho, 2)

    return float('{0:.2f}'.format(ortho))


def median(p1, p2):
    """
    median coordinates of two points.
    p1 and p2 must both be arrays of x and y coordinates.
    returns an array of x and y coordinates.
    """
    med = p1+(p2-p1)/2.
    return med

