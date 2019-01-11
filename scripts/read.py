#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def fspex(fname):
    """
    get file specs
    --------------
    inputs:
        fname (str):
            full path to file name

    outputs:
        hdr (list of str):
            header, i.e. 1st line of 'fname', as list of strings
        fsz (int):
            number of lines in 'fname'
    """
    with open(fname) as f:
        cnt = f.readlines()
    f.close()

    hdr = cnt[0].strip('\n').split(';')
    fsz = len(cnt)-1

    return hdr, fsz


def fread(fname):
    """
    read file contents
    --------------
    inputs:
        fname (str):
            full path to file name

    outputs:
        data (ndarray):
            file contents aranged in a rank 2 array
            first dim: lines
            second dim: columns
    """
    data=[]
    with open(fname) as fnl:
        for line in fnl:
            data.append(line.strip('\n').split(';'))

    fnl.close()

    y = np.asarray([float(data[s][0].split(', ')[0]) for s in range(1,len(data))])
    x = np.asarray([float(data[s][0].split(', ')[1]) for s in range(1,len(data))])
    m2 = np.asarray([float(data[s][1]) for s in range(1,len(data))])
    nbpl = np.asarray([float(data[s][2]) for s in range(1,len(data))])
    m2pltot = np.asarray([float(data[s][3]) for s in range(1,len(data))])

    data = np.vstack((x, y, m2, nbpl, m2pltot))

    return data
