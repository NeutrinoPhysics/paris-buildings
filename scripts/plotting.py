#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 

from tools import clusam

def plot_clusters(clu, csa, cns, fsav):
	"""
	clu 	: 	x,y array of clusters
	csa 	: 	x,y array of samples
	cns 	: 	cluster number id 
	fsav 	: 	save file path
	"""

	nk = clu.shape[-1]
	cns = list(map(clusam, np.arange(nk)))

	fig=plt.figure(figsize=[40,30])
	for i in range(nk):
		plt.scatter(x=csa[0,cns[i]], y=csa[1,cns[i]], alpha=0.1, s=1)
	plt.scatter(x=clu[0], y=clu[1], color='k', marker='*', s=50)
	plt.axis('off')
	plt.savefig(fsav)
	plt.close()



