import numpy as np
import h5py
from imageio import imread
from PIL import Image
from tqdm import tqdm
from math import ceil
import cv2

import sys
if not sys.version_info > (3,0):
	from SyntheticTools import file_io

def imresize(arr,size):
	arr = 255*arr
	return np.array(Image.fromarray(arr.astype(np.uint8)).resize((size[1],size[0])))

def loadLightField(filename):

	# Attempt to find filetype
	if filename[-9:] == "_eslf.png":
		print("File Type: ESLF")
		filetype = "eslf"
	elif filename[-4:] == ".mat":
		print("File Type: MAT")
		filetype = "MAT"
	elif filename[-9:] == ".eslf.png":
		print("File Type: New ESLF")
		filetype = "New ESLF"
	else:
		print("Light Field file type not found, assuming Synthetic Lightfield")
		filetype = "Synthetic"

	if filetype == "eslf":
		name = filename[:-9]
		meta = name + ".json"
		lf = loadESLF(filename,meta)

	elif filetype == "MAT":
		lf = loadMAT(filename)

	elif filetype == "New ESLF":
		name = filename[:-9]
		meta = name + ".json"
		lf= loadESLF(filename,meta,version=2)

	elif filetype == "Synthetic":
		print("You will need to presave the lightfield using Python 2.7")
		lf = np.load(filename)
		
	return lf
	
def loadLightFieldWithDepth(filename):

	from DepthFunctions import loadDepth
	lf = loadLightField(filename)
	
	# Attempt to find filetype
	if filename[-9:] == "_eslf.png":
		filetype = "eslf"
	elif filename[-4:] == ".mat":
		filetype = "MAT"
	elif filename[-9:] == ".eslf.png":
		filetype = "New ESLF"
	else:
		filetype = "Synthetic"

	if filetype == "eslf":
		name = filename[:-9]
		depth_name = name+ "_warp_depth.png"
		depth = loadDepth(depth_name)

	elif filetype == "MAT":
		name = filename[:-4]
		depth_name = name + ".png"
		depth = loadDepth(depth_name)
		im_r,im_c,_,_,_ = lf.shape
		rows,cols = depth.shape

		lightfield = np.zeros((im_r,im_c,rows,cols,3))

		for r in range(im_r):
			for c in range(im_c):
				lightfield[r,c] = imresize(lf[r,c,:,:,:3],(rows,cols))/255.0
				
		lf = lightfield

	elif filetype == "New ESLF":
		name = filename[:-9]
		depth_name = name+ ".depth.png"
		depth = loadDepth(depth_name)

	elif filetype == "Synthetic":
		print("You will need to presave the lightfield using Python 2.7")
		depth = np.load(filename[:-4]+".npy")
		
	return lf, depth	

def loadMAT(filename):

	if not sys.version_info > (3,0):
		print("Requires Python 3")
		return

	# Read the Light Field Data from the .mat file
	lightfield_data = {}
	f = h5py.File(filename)
	for k, v in f.items():
		lightfield_data[k] = np.array(v)
		
	# Fields in the lightfield_data dictionary
	# DecodeOptions
	# GeneratedByInfo
	# LF (the actual data)
	# LensletGridModel
	# RectOptions
	# WhiteImageMetaData
		
	# Extract the light field data
	lightfield = lightfield_data["LF"]

	# Reorder the axes
	lightfield = np.transpose(lightfield,(4,3,2,1,0))
		
	return lightfield/np.amax(lightfield)
	
def colorCorrection(image, ccm, black_level = [0.0, 0.0, 0.0], white_level = [1.0,1.0,1.0], balance = [1.0,1.0,1.0], gamma = 1.0):
	
	# ccm is the Color Correction Matrix
	rows, cols, _ = image.shape
	black_level = np.array(black_level)
	white_level = np.array(white_level)
	ccm = np.array(ccm)
	balance = np.array(balance)
	
	# Flatten input to a flat list of RGBs
	result = np.reshape(image[:,:,:3], (rows*cols,3))
	
	# Apply levels and devignetting
	result = (result - (black_level/white_level))/(1-(black_level/white_level))
	#if np.any(devignette):
		#devignette = np.reshape(devignette, (rows*cols,1))
		#devignette = (devignette - (black_level/white_level))/(1-(black_level/white_level))
		#image = image/devignette
	
	# Apply balance and color correction matrix
	result *= balance
	result = np.matmul(result,ccm)

	# Reshape to original size
	result = np.reshape(result, (rows,cols, 3))

	# Clip values
	result = np.minimum(1,np.maximum(0, result))
	
	# Saturating eliminates some issues with clipped pixels, but is aggressive and loses information
	#saturation = np.matmul(balance,ccm)
	#saturation = np.amin(saturation)
	#result = np.minimum(saturation, np.maximum(0,result))/saturation
	
	# Apply gamma
	result = np.power(result, gamma)
	
	# Normalize image
	result = result/np.amax(result)
	
	# Reappend the alpha channel
	# result = np.concatenate((result, image[:,:,[3]]),axis=2)
	
	return result
	
def retrieveColorInfoV1(filename):
	
	import json
	data = json.load(open(filename))
	
	ccm = data["frames"][0]["frame"]["metadata"]["image"]["color"]["ccm"]
	#balance = data["frames"][0]["frame"]["metadata"]["image"]["color"]["whiteBalanceGain"]
	black_level =  data["frames"][0]["frame"]["metadata"]["image"]["pixelFormat"]["black"]
	white_level =  data["frames"][0]["frame"]["metadata"]["image"]["pixelFormat"]["white"]
	
	ccm = np.reshape(np.array(ccm),(3,3)).T
	#balance = np.array([balance["r"], balance["gb"], balance["b"]])
	black_level = np.array([black_level["r"], black_level["gb"], black_level["b"]])
	white_level = np.array([white_level["r"], white_level["gb"], white_level["b"]])
		
	return ccm, black_level, white_level
	
def retrieveColorInfoV2(filename):
	
	import json
	data = json.load(open(filename))
	
	ccm = data["image"]["color"]["ccm"]
	#balance = data["image"]["color"]["whiteBalanceGain"]
	black_level =  data["image"]["pixelFormat"]["black"]
	white_level =  data["image"]["pixelFormat"]["white"]
	
	ccm = np.reshape(np.array(ccm),(3,3)).T
	#balance = np.array([balance["r"], balance["gb"], balance["b"]])
	black_level = np.array([black_level["r"], black_level["gb"], black_level["b"]])
	white_level = np.array([white_level["r"], white_level["gb"], white_level["b"]])
		
	return ccm, black_level, white_level
	
def loadESLF(filename, meta=None, version = 1, devignette=None, w=14, h=14):

	# Load ESLF image
	image = imread(filename).astype(np.float32)
	image = image/np.amax(image)
	
	# If metadata was provided...
	if meta:
		if version == 1:
			# Retrieve Color Correction Matrix from metadata
			ccm, black_level, white_level = retrieveColorInfoV1(meta)
		elif version == 2:
			# Retrieve Color Correction Matrix from metadata
			ccm, black_level, white_level = retrieveColorInfoV2(meta)
		
		# Perform Color Correction
		image = colorCorrection(image, ccm, black_level, white_level)#, devignette)
		
	else:
		print("No Color Correction performed")
	
	# Find it's physical size
	rows, cols, ch = np.shape(image)
	im_h = int(rows/h)
	im_w = int(cols/w)
	
	# Generate the appeture dimensions
	lightfield = np.reshape(image,(im_h,h,im_w,w,ch))
	
	# Reorder the axes
	lightfield = np.transpose(lightfield,(1,3,0,2,4))
	
	return lightfield
	
	
def loadViews(foldername, w=17, h=17):

	import os
	files = os.listdir(foldername)
	images = []
	for f in files:
		images.append(imread(foldername + f).astype(np.float32))
	
	# Reshape to the lightfield dimesions
	images = np.array(images)
	_, rows, cols, ch = images.shape
	lightfield = np.reshape(images, (h,w,rows,cols,ch))/255.0
	
	# Append alpha channel if not present
	if ch == 3:
		alpha = np.ones((h,w,rows,cols,1))
		lightfield = np.concatenate((lightfield,alpha),axis=4)
	
	return lightfield

def loadViewsNumbered(foldername, w=17, h=17, desc=""):

	import os
	files = os.listdir(foldername)
	images = []
	
	# Get Naming Convention
	f = files[0]
	name = f.split("_")[0]
	
	# Get files in order
	for i in range(-h//2+1, h//2+1):
		for j in range(-w//2+1, w//2+1):
			images.append(imread(foldername+name+"_"+str(i)+"_"+str(j)+desc+".png").astype(np.float32))
	
	# Reshape to the lightfield dimesions
	images = np.array(images)
	_, rows, cols, ch = images.shape
	lightfield = np.reshape(images, (h,w,rows,cols,ch))/255.0
	
	# Append alpha channel if not present
	if ch == 3:
		alpha = np.ones((h,w,rows,cols,1))
		lightfield = np.concatenate((lightfield,alpha),axis=4)
	
	return lightfield
	
	
def loadNPY(filename):
	return np.load(filename)
	
	
# Note this assumes the folders are labeled in ascending order and there are no other files in the directory
def loadFocalStack(directory):
	
	images = []
	
	import os
	files = os.listdir(directory)
	for file in tqdm(files):
		im = imread(directory + file).astype(np.float32)
		images.append(im)
	
	images = np.array(images)/255.0
	
	focalstack = np.array(images)
	
	return focalstack
	

def loadFocalStacks(directory, r=375,c=540):
	
	images = []
	
	import os
	folders = os.listdir(directory)
	for folder in folders:
		files = os.listdir(directory + "\\" + folder)
		for file in tqdm(files):
			im = imread(directory + "\\" + folder + "\\" + file).astype(np.float32)[:r,:c]
			images.append(im)
	
	images = np.array(images)/255.0
	
	print(images.shape)
	
	_,rows,cols,ch = images.shape
	
	# Reshape to associate images of the same focal stack
	focalstacks = np.reshape(images,(size(files),size(folders),rows,cols,ch),order="F")
	
	return focalstacks
	
def loadSyntheticLF(foldername, gantry = False):
	
	lightfield = file_io.read_lightfield(foldername)
	lightfield = lightfield.astype(np.float32)/255
	
	if gantry:
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
	
		print(offset)
	
		h, w, im_h, im_w, ch = lightfield.shape
		im_h = im_h - (h-1)*offset
		im_w = im_w - (w-1)*offset
		gantry_lightfield = np.zeros((h,w,im_h,im_w,ch))
		
		for i in range(h):
			for j in range(w):
				gantry_lightfield[i,j] = lightfield[i,j,i*offset:i*offset+im_h,j*offset:j*offset+im_w]
				
		lightfield = gantry_lightfield
	
	return lightfield




