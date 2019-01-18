#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- import packages
import os
import numpy as np
#import scipy as sp
#from scipy import stats, ndimage, signal
import matplotlib as mpl 
#import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

# --- load modules
from read import fspex, fread
import tools as tools
from plotting import *



def find_roads(data):

	print('finding roads...')
	coords = data if data.shape[0]==2 else data[:2,:]

	# --- scatter plot location of buildings
	map_buildings(	coords = coords,
					fsav   = os.path.join(pdir, 'paris_coords_light.png'),
					style  = 'light',
					fgsz   = [20,16],
					show   = False
			 	)

	# --- save a thicker version for processing
	map_buildings(	coords = coords,
					fsav   = os.path.join(pdir, 'paris_coords_thick.png'),
					style  = 'thick',
					fgsz   = [40,30],
					show   = False
			 	 )



	# --- load map from last image
	img = mpimg.imread(os.path.join(pdir, 'paris_coords_thick.png'))
	img = img[:,:,:-1].mean(axis=-1)

	# --- high pass filter
	hpf = tools.highpass(img=img)

	# --- threshold value
	thr = 100

	histo_hpf(hpf   = hpf,
			  thr   = thr,
			  edge  = 256,
			  fsav  = os.path.join(pdir, 'paris_hpflat.png'),
			  show  = False
			  )


	# --- select pixels above threshold value to make contour map
	msk = tools.mask(hpf=hpf, thr=thr)

	# --- plot countours map
	map_mask( msk  = msk,
			  fsav = os.path.join(pdir, 'paris_roads.png'),
			  show = False 
			 )

	# --- make 2d histogram  for density
	map_density(coords  = coords,
				palette = 'CMRmap',
				fgsz    = [40,30],
				fsav    = os.path.join(pdir, 'paris_density.png'),
				show    = False
				)
	print('done\n')

	return





def launch_cluster(csa, nk, mxt):

	print('clustering...')
	# === meta params ===

	nk  = nk if nk else 15		# number of clusters
	stp = 0						# initial step number (set to zero)
	mxt = mxt if mxt else 20	# max step

	# --- define output directories
	bas = 'k'+str(nk)
	kpdir = os.path.join(pdir, bas) # kmeans plots directory
	kfdir = os.path.join(fdir, bas) # kmeans files directory
	fmuout = os.path.join(kfdir, bas+'_mu.npy')
	fidout = os.path.join(kfdir, bas+'_id.npy')

	# --- create directories if they don't exist
	os.path.exists(kpdir) if os.path.exists(kpdir) else os.makedirs(kpdir)
	os.path.exists(kfdir) if os.path.exists(kfdir) else os.makedirs(kfdir)



	# === initiation step ===

	# --- randomly draw first clusters
	drw = np.random.randint(low=0, high=fsz-1, size=nk, dtype=int)
	clu = csa[:,drw]

	# --- first iteration
	print("step 0")
	kmc = tools.kmeans(dat=csa, clu=clu)
	cmu_arr = kmc.cmu.T.copy()
	cid_arr = kmc.cid.copy()

	# --- plot clusters map
	fsav = os.path.join(kpdir, 'k'+str(nk)+'_step_0.png')
	plot_clusters(clu=kmc.clu, csa=kmc.csa, cns=kmc.cns, fsav=fsav)

	# === interation steps ===

	while ( (np.not_equal(kmc.cmu.T, kmc.clu).sum()>0) and (stp < mxt) ):

		stp += 1
		print("step {}".format(stp))
		fsav = os.path.join(kpdir, 'k'+str(nk)+'_step_'+str(stp)+'.png')
		fidout = os.path.join(kfdir, bas+'_id_step'+str(stp)+'.npy')

		new_clu = kmc.cmu.T.copy()
		kmc = None
		kmc = tools.kmeans(dat=csa, clu=new_clu)
		cmu_arr = np.dstack((cmu_arr, kmc.cmu.T))
		cid_arr = np.vstack((cid_arr, kmc.cid))

		# --- plot clusters map
		plot_clusters(clu=kmc.cmu.T, csa=kmc.csa, cns=kmc.cns, fsav=fsav)

	# --- dump into file
	np.save(fmuout, cmu_arr)
	np.save(fidout, cid_arr)

	print('done.')

	return





if __name__ == '__main__':

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


	# ---------------------------------------
	# /!\ comment / uncomment following lines 
	#     depending on want you want done.
	# ---------------------------------------

	find_roads(data=data)
	launch_cluster(csa=data[:2,:], nk=300, mxt=20)

