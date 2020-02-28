import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from scipy.misc import imresize, imsave

def plotImage(im, title=""):
	# Scale appropriately
    scaled = np.multiply(im,255/np.max(im))
	
    im = np.array(scaled[:,:,0:3], dtype=np.uint8)
    plt.imshow(im, vmin = 0, vmax = 255)
    plt.title(title)
    plt.show()	
	
def getOriginal(lf):
	
	h, w, im_h, im_w, ch = lf.shape
	
	#How to deal with alpha channel?
	#print("Does not account for alpha channel")
	
	return np.mean(lf,axis=(0,1))
	
def getESLF(lf):

	h, w, im_h, im_w, ch = lf.shape
	
	eslf = np.transpose(lf, (2,0,3,1,4))
	
	return np.reshape(eslf, (im_h*h, im_w*w,ch))

def getViews(lf):

	h, w, im_h, im_w, ch = lf.shape
	
	views = np.transpose(lf, (0,2,1,3,4))
	
	return np.reshape(views, (im_h*h, im_w*w,ch))
	
def getView(lf,ic,jc):

	h, w, _, _, ch = lf.shape
	
	i = (h-1)//2 + ic
	j = (w-1)//2 + jc
	
	return lf[i,j,:,:,:3]

def getFocalStack(lf, near=0.5, far=2.0, near_step = 0.1, far_step=.2):
	
	images = []
	
	near_list = list(np.arange(near,1.0,near_step))
	far_list = list(np.arange(1.0,far+far_step,far_step))
	
	for alpha in tqdm(near_list+far_list):
		im = refocus(lf,alpha)
		scaled = np.multiply(im,255/np.max(im))
		result = np.array(scaled[:,:,0:3], dtype=np.uint8)
		images.append(result)
			
	return np.array(images), near_list+far_list

def padLF(lf):

	from math import floor

	h, w, _, _, _ = lf.shape
	
	ypad = floor(h/2)
	xpad = floor(w/2)
	
	return np.pad(lf,((0,0),(0,0),(ypad,ypad+1),(xpad,xpad+1),(0,0)),'edge')
	
def clipLF(lf,num_rows,num_cols, end_only = False):

	if end_only:
		return lf[:,:,0:-num_rows-1,0:-num_cols-1,:]
	else:
		return lf[:,:,num_rows:-num_rows-1,num_cols:-num_cols-1,:]
	
def refocus(lf,alpha=1.0):
	
	from math import floor
	from scipy.interpolate import RectBivariateSpline
	h, w, im_h, im_w, _ = lf.shape

	x, y = np.array(range(im_w)),np.array(range(im_h))
	
	subimage = np.zeros((im_h,im_w,3))
	
	result = np.zeros((im_h,im_w,3))
	
	# Sub along all subapeture images
	for u in range(w):
		for v in range(h):	
			
			# Associate each view with a distance from the center
			vc = v - h/2 + .5
			uc = u - w/2 + .5
			
			for k in range(3):
				# Make smooth grid to sample from
				ip = RectBivariateSpline(y, x, lf[v,u,:,:,k], kx=1, ky=1)
				subimage[:,:,k] += ip(y+vc*(1 - 1.0/alpha), x+uc*(1 - 1.0/alpha))
				
			result += (1.0/(w*h)) * subimage
			
	# Normalize
	result *= (1.0/(w*h))
	
	return result
	
def makeAnimation(lf,option="Horizontal"):

	h, w, im_h, im_w, ch = lf.shape
	
	images=[]
	indices=[]
	
	if option == "Refocus":
		
		near = list(np.arange(.5,1.0,.1))
		far = list(np.arange(1.0,2.0,.1))
		
		for alpha in tqdm(near+far):
			im = refocus(lf,alpha)
			scaled = np.multiply(im,255/np.max(im))
			result = np.array(scaled[:,:,0:3], dtype=np.uint8)
			images.append(result)
			
		return images + images[::-1]
	
	elif option == "Horizontal":
	
		i = int(h/2)
	
		# Trace Forward
		for j in range(w):
			indices.append((i,j))
			
		# Trace Backward
		for j in range(1,w-1)[::-1]:
			indices.append((i,j))
			
	elif option == "Vertical":
	
		j = int(w/2)
	
		# Trace Forward
		for i in range(h):
			indices.append((i,j))
			
		# Trace Backward
		for i in range(1,h-1)[::-1]:
			indices.append((i,j))
	
	elif option == "Rotation":
	
		from math import pi, cos, sin
		num_angles = 20
		angles = [2*pi*x/num_angles for x in range(num_angles)]
		
		radius = min(w,h)/2 - 1
		
		for angle in angles:
			i = round(radius + radius*sin(angle))
			j = round(radius + radius*cos(angle))
			indices.append((i,j))
	
	else:
		print("Option parameter did not match available options")
	
	for i,j in indices:
		i = int(i); j = int(j)
		im = lf[i,j]
		scaled = np.multiply(im,255/np.max(im))
		result = np.array(scaled[:,:,0:3], dtype=np.uint8)
		images.append(result)
		
	return images
	
def plotAnimation(images, title=""):
	fig = plt.figure()
	ims = []
	for im in images:
		result = plt.imshow(im, vmin=0, vmax=255, animated=True)
		ims.append([result])

	ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
	plt.show()
