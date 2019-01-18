#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

import scipy as sp
from scipy import stats, ndimage, signal



def plot_clusters(clu, csa, cns, fsav):
	"""
	clu 	: 	x,y array of clusters
	csa 	: 	x,y array of samples
	cns 	: 	cluster number id 
	fsav 	: 	save file path
	"""

	nk = clu.shape[-1]

	fig=plt.figure(figsize=[40,30])
	for i in range(nk):
		plt.scatter(x=csa[0,cns[i]], y=csa[1,cns[i]], alpha=0.4, s=1)
	plt.scatter(x=clu[0], y=clu[1], color='k', marker='s', s=50)
	plt.axis('off')
	plt.savefig(fsav)
	plt.close()

	return




def map_buildings(coords, fsav, style, fgsz, show):
	"""
	scatter plot location of buildings
	"""

	fgsz  = fgsz if fgsz else [20,16]

	al = 0.1
	ms = 1
	if style=='thick':
		al *= 10
		ms *= 10

	fig=plt.figure(figsize=fgsz)
	plt.scatter(coords[0], coords[1], color='k', alpha=al, s=ms)
	plt.axis('off')

	if show:
		plt.show()
	else:
		plt.savefig(fsav)
		plt.close()

	return




def histo_hpf(hpf, thr, edge, fsav, show):

	# --- define histogram bins and their edges
	edge = edge if edge else 256
	beans = np.arange(0, edge)
	beanedge = np.hstack((beans, beans[-1]+1)) - 0.5


	fig=plt.figure(figsize=[8,6])
	hst = plt.hist(hpf.flatten(), bins=beanedge, align='mid', log=True, histtype='step', color='k', linewidth=1.5)[0]
	
	# --- set figure limits
	ymin, ymax = plt.ylim()
	lyin, lyax = np.log10(ymin), np.log10(ymax)
	plt.vlines(thr, ymin, ymax, color='r', linewidth=0.5)
	plt.ylim(ymin, ymax)
	plt.xlim(beans[0], beans[-1])
	xmin, xmax = plt.xlim()
	
	plt.xlabel('pixel value', fontsize=14)
	plt.title('histogram of flattened high pass filter map', fontsize=12, loc='left')
	plt.fill_between(x=beans[thr:], y1=ymin, y2=hst[thr:], color='r', alpha=0.25)

	pct = hst[thr:].sum()/hst.sum()
	plt.annotate(str(np.round(100.*(1.-pct), 2))+'%', xy=(xmin+0.33*(thr-xmin), 10**(lyin+0.2*(lyax-lyin))), color='darkred', fontsize=12)
	plt.annotate(str(np.round(100.*pct, 2))+'%', xy=(thr+0.33*(xmax-thr), 10**(lyin+0.2*(lyax-lyin))), color='darkred', fontsize=12)
	
	if show:
		plt.show()
	else:
		plt.savefig(fsav)
		plt.close()

	return





def map_mask(msk, fsav, show):

	fig=plt.figure(figsize=[40,30])
	plt.imshow(msk, cmap=mpl.cm.binary, interpolation=None)
	plt.ylim(2600, 340)
	plt.xlim(610, 3500)
	plt.axis('off')
	if show:
		plt.show()
	else:
		plt.savefig(fsav)
		plt.close()

	return




def map_density(coords, palette, fgsz, fsav, show):

	# --- make 2d histogram  for density
	h = plt.hist2d(coords[0], coords[1], bins=(1000, 750), cmax=7)[0]
	plt.close()

	# --- define smoothing kernel
	gs = signal.gaussian(M=3, std=0.5)
	kernel = np.outer(gs, gs)
	nanx, nany = np.where(np.isnan(h)) # some 'nan' erros. where are they? 
	h[nanx, nany] = 8.                 # convert those with arbitrry value

	# --- convolve density histogram with gaussian kernel to smooth it
	gh = sp.signal.convolve2d(h.T, kernel, mode='full', fillvalue=0)

	# --- define color palette
	palette = palette if palette else 'CMRmap'
	my_cmap = mpl.cm.get_cmap(palette)
	my_cmap.set_under('w')

	# --- plot density plot
	fgsz = fgsz if fgsz else [40,30]
	fig = plt.figure(figsize=fgsz)
	plt.imshow(gh, origin='lower', cmap=my_cmap, vmin=0.001)
	plt.axis('off')
	#plt.colorbar(orientation='horizontal', shrink=0.3, aspect=50, pad=0.15)

	if show:
		plt.show()
	else:
		plt.savefig(fsav)
		plt.close()

	return

