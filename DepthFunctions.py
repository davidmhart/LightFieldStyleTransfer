from imageio import imsave, imread
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import matplotlib.pyplot as plt
import numpy as np
from math import ceil

import sys
if not sys.version_info > (3,0):
	from SyntheticTools import file_io


def loadDepth(filename, meta=None):
	
	depth = imread(filename).astype(np.float32)
    
	#Scaling, assume farthest point is maximum
	depth = depth/np.max(depth)
	
	return depth
	
def loadSyntheticDepth(foldername, gantry = False):
	
	depth_map = file_io.read_depth(foldername, highres=False)
	
	if gantry:
		lightfield = file_io.read_lightfield(foldername)
		params = file_io.read_parameters(foldername)
		
		baseline_mm = params["baseline_mm"]
		focal_length_mm = params["focal_length_mm"]
		focus_dist_m = params["focus_distance_m"]
		sensor_mm = params["sensor_size_mm"]
		width = params["width"]
		height = params["height"]
		
		offset = baseline_mm * focal_length_mm / focus_dist_m / 1000. / sensor_mm * max(width, height)
	
		# Round, interpolate in future?
		offset = int(ceil(offset))
		
		h, w, im_h, im_w, ch = lightfield.shape
		im_h = im_h - (h-1)*offset
		im_w = im_w - (w-1)*offset
		
		i = h//2
		j = w//2
		
		depth_map = depth_map[i*offset:i*offset+im_h,j*offset:j*offset+im_w]
	
	return depth_map
	
def loadSyntheticDisparity(foldername, gantry = False):
	
	disparity = file_io.read_disparity(foldername, highres=False)
	#disparity = file_io.read_pfm(foldername + "gt_disp_lowres.pfm")
	
	if gantry:
		params = file_io.read_parameters(foldername)
		lightfield = file_io.read_lightfield(foldername)
		
		baseline_mm = params["baseline_mm"]
		focal_length_mm = params["focal_length_mm"]
		focus_dist_m = params["focus_distance_m"]
		sensor_mm = params["sensor_size_mm"]
		width = params["width"]
		height = params["height"]
		
		offset = baseline_mm * focal_length_mm / focus_dist_m / 1000. / sensor_mm * max(width, height)
	
		# Round, interpolate in future?
		offset = int(ceil(offset))
		
		h, w, im_h, im_w, ch = lightfield.shape
		im_h = im_h - (h-1)*offset
		im_w = im_w - (w-1)*offset
		
		i = h//2
		j = w//2
		
		disparity = disparity[i*offset:i*offset+im_h,j*offset:j*offset+im_w]
	
		disparity += offset
	
	return disparity
	
def plotDepth(depth):
	
	#Scaling, assume farthest point is maximum
	depth = depth/np.max(depth)
	
	plt.imshow(depth,vmin=0,vmax=1,cmap="Greys_r")
	plt.show()
	
def plotDisp(disp):
	
	plt.imshow(disp,cmap="Greys_r")
	plt.show()
	
			
	
	