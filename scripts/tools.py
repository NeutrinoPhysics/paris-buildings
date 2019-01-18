#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import functools as ft
import scipy as sp
from scipy import stats, ndimage, signal

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



def highpass(img):
    # --- high pass filter          
    hpf = img - sp.ndimage.gaussian_filter(img, 2)

    # --- convert to unsigned integer
    hpf -= hpf.min()
    hpf /= hpf.max()
    hpf *= 255
    hpf  = hpf.astype(np.uint8)

    return hpf


def mask(hpf, thr):
    # --- select pixels above threshold value to make contour map
    return sp.ndimage.filters.median_filter((hpf >= thr).astype(float), 2).astype(np.uint8)



class kmeans(object):

    def __init__(self, dat, clu):
        """
        nk   :  nuber of clusters
        csa  :  x,y coordinates array of samples
        clu  :  x,y coordinates array of clusters
        """

        self.csa = dat
        self.sz = self.csa.shape[-1]

        self.clu = clu
        self.nk = self.clu.shape[-1]

        self.cid = np.asarray(list(map(self.clustid, np.arange(self.sz))))
        self.cns = list(map(self.clusam, np.arange(self.nk)))
        self.cmu = np.asarray(list(map(self.cmean,np.arange(self.nk))))

        return 



    def dtc(self, cn, sa):
        """
        distance to cluster
        sa (int)    :   sample number
        cn (int)    :   cluster number 
        """
        return distance(p1=self.csa[:,sa], p2=self.clu[:,cn])


    def clustid(self, i):
        """
        identify id of cluster from sample number
        """
        return np.asarray(list(map(ft.partial(self.dtc,sa=i), np.arange(self.nk)))).argmin()



    def clusam(self, cn):
        """
        cluster number sample
        cn (int)    :   cluster number
        """
        return np.where(self.cid==cn)[0]


    def cmean(self, cn):
        """
        cluster mean
        """
        return np.around(self.csa[:,self.cns[cn]].mean(axis=1),6)


