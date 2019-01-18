#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- import packages
import os
import numpy as np
import scipy as sp
from scipy import stats, ndimage, signal
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

# --- load modules
from read import fspex, fread
import tools as tools



# --- directories definition
home = os.getcwd().replace('/scripts', '')  # main directory
sdir = os.path.join(home, 'scripts')        # scripts directory
fdir = os.path.join(home, 'files')          # files directory
pdir = os.path.join(home, 'plots')          # plots directory


# --- load data from file
paris = os.path.join(fdir, 'paris_buildings.csv')

# --- get specs: header, file size, data 
hdr, fsz = fspex(paris)     # see 'read.py' for details
data = fread(paris)         # see 'read.py' for details

"""
# --- scatter plot location of buildings
fig=plt.figure(figsize=[20,16])
plt.scatter(data[0], data[1], color='k', alpha=0.1, s=1)
plt.axis('off')
plt.savefig(os.path.join(pdir, 'paris_coords_light.png'))
plt.close()







# === IDENTIFY ROADS ===

# --- save a thicker version for processing
fig=plt.figure(figsize=[40,30])
plt.scatter(data[0], data[1], color='k', alpha=1, s=10)
plt.axis('off')
plt.savefig(os.path.join(pdir, 'paris_coords_thick.png'))
plt.close()


# --- load map from last image
img = mpimg.imread(os.path.join(pdir, 'paris_coords_thick.png'))
img = img[:,:,:-1].mean(axis=-1)

# --- high pass filter          
hpf = img - sp.ndimage.gaussian_filter(img, 2)

# --- convert to unsigned integer
hpf-=hpf.min()
hpf/=hpf.max()
hpf*=255
hpf=hpf.astype(np.uint8)

# --- define histogram bins and their edges
beans = np.arange(0, 256)
beanedge = np.hstack((beans, 256)) - 0.5

# --- choose cutoff
thr=100

# --- plot histogram of high pass filter
fig=plt.figure(figsize=[8,6])
hst = plt.hist(hpf.flatten(), bins=beanedge, align='mid', log=True, histtype='step', color='k', linewidth=1.5)[0]
ymin, ymax =plt.ylim()
lyin, lyax = np.log10(ymin), np.log10(ymax)
plt.vlines(thr, ymin, ymax, color='r', linewidth=0.5)
plt.ylim(ymin, ymax)
plt.xlim(0, 255)
xmin, xmax = plt.xlim()
plt.xlabel('pixel value', fontsize=14)
plt.title('histogram of flattened high pass filter map', fontsize=12, loc='left')
plt.fill_between(x=beans[thr:], y1=ymin, y2=hst[thr:], color='r', alpha=0.25)
pct = hst[thr:].sum()/hst.sum()
plt.annotate(str(np.round(100.*(1.-pct), 2))+'%', xy=(xmin+0.33*(thr-xmin), 10**(lyin+0.2*(lyax-lyin))), color='darkred', fontsize=12)
plt.annotate(str(np.round(100.*pct, 2))+'%', xy=(thr+0.33*(xmax-thr), 10**(lyin+0.2*(lyax-lyin))), color='darkred', fontsize=12)
plt.savefig(os.path.join(pdir, 'paris_hpflat.png'))
plt.close()

# --- select pixels above threshold value to make contour map
msk = sp.ndimage.filters.median_filter((hpf >= thr).astype(float), 2).astype(np.uint8)

# --- plot countours map
fig=plt.figure(figsize=[40,30])
plt.imshow(msk, cmap=mpl.cm.binary, interpolation=None)
plt.ylim(2600, 340)
plt.xlim(610, 3500)
plt.axis('off')
plt.savefig(os.path.join(pdir, 'paris_roads.png'))
plt.close()











# === DENSITY PLOT ===

# --- make 2d histogram  for density
h = plt.hist2d(data[0], data[1], bins=(1000, 750), cmax=7)[0]
plt.close()

# --- define smoothing kernel
gs = signal.gaussian(M=3, std=0.5)
kernel = np.outer(gs, gs)
nanx, nany = np.where(np.isnan(h)) # some 'nan' erros. where are they? 
h[nanx, nany] = 8.                 # convert those with arbitrry value

# --- convolve density histogram with gaussian kernel to smooth it
gh = sp.signal.convolve2d(h.T, kernel, mode='full', fillvalue=0)

# --- define color palette
my_cmap = mpl.cm.get_cmap('CMRmap')
my_cmap.set_under('w')

# --- plot density plot
fig = plt.figure(figsize=[40,30])
plt.imshow(gh, origin='lower', cmap=my_cmap, vmin=0.001)
plt.axis('off')
#plt.colorbar(orientation='horizontal', shrink=0.3, aspect=50, pad=0.15)
plt.savefig(os.path.join(pdir, 'paris_density.png'))
plt.close()

"""








# ======= K MEANS ALGORITHM ========



# === meta params ===

nk = 60		# number of clusters
stp = 0		# initial step number (set to zero) 

# --- define output directories
bas = 'k'+str(nk)
kpdir = os.path.join(pdir, bas) # kmeans plots directory
kfdir = os.path.join(fdir, bas) # kmeans files directory
fmuout = os.path.join(kfdir, bas+'_mu.npy')

# --- create directories if they don't exist
os.path.exists(kpdir) if os.path.exists(kpdir) else os.makedirs(kpdir)
os.path.exists(kfdir) if os.path.exists(kfdir) else os.makedirs(kfdir)



# === initiation step ===

# --- isolate x,y coordinates from full data set
csa = data[:2,:]

# --- randomly draw first clusters
drw = np.random.randint(low=0, high=fsz-1, size=nk, dtype=int)
clu = csa[:,drw]

# --- first iteration
cid, cmu = tools.kmeans(dat=csa, clu=clu)
cmu_arr = cmu.copy()


# === interation steps ===

while(np.not_equal(cmu, clu).sum()>0):

	stp += 1
	print("step {}".format(stp))
	fsav = os.path.join(kpdir, 'k'+str(nk)+'_step_'+str(stp)+'.png')

	cid, cmu_new = tools.kmeans(dat=csa, clu=cmu)
	
	cmu_arr = np.dstack((cmu_arr, cmu_new))
	cmu = cmu_new.copy()
	cmu_new = None

	cid_arr = np.vstack((cid_arr, cid))
	cid = cid_new.copy()
	cid_new = None

	cns = list(map(tools.clusam, np.arange(nk)))

	# --- dump into file
	np.save(fidout, np.asarray(cid))

	# --- plot clusters map
	plot_cluster(clu=clu, csa=csa, cns=cns, fsav=fsav)



# --- dump into file
np.save(fmuout, cmu_arr)
np.save(fidout, np.asarray(cid))
