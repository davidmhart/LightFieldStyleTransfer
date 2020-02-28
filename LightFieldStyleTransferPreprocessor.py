from imageio import imwrite
from scipy.misc import imresize
import numpy as np
from tqdm import tqdm
from LoadLightField import *
from LightFieldFunctions import *
from SaveLightField import *
from DepthFunctions import *
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm, trange
from time import time

import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

        
def calcSimilarity(xv,yv,dx,dy,ipR,ipG,ipB,image):
    r = ipR(yv-dy,xv-dx,grid=False)
    g = ipG(yv-dy,xv-dx,grid=False)
    b = ipB(yv-dy,xv-dx,grid=False)
    
	#Normalized RGB Distance
    diff = (image[:,:,0]-r)**2 + (image[:,:,1]-g)**2 + (image[:,:,2]-b)**2
    diff = np.sqrt(diff)/np.sqrt(3)
    return 1-diff


def warpViewNP(content,dx,dy):

    rows, cols, _ = content.shape

    # Generate interpolated field
    x, y = np.array(range(cols)),np.array(range(rows))
    xv, yv = np.meshgrid(x,y)

    # Warp left image to the right side view
    ipR = RectBivariateSpline(y, x, content[:,:,0], kx=1, ky=1)
    ipG = RectBivariateSpline(y, x, content[:,:,1], kx=1, ky=1)
    ipB = RectBivariateSpline(y, x, content[:,:,2], kx=1, ky=1)

    warped = np.zeros((rows,cols,_))
    warped[:,:,0] = ipR(yv+dy,xv+dx,grid=False)
    warped[:,:,1] = ipG(yv+dy,xv+dx,grid=False)
    warped[:,:,2] = ipB(yv+dy,xv+dx,grid=False)

    return warped


def resizeNP(data, shape):

    rows, cols = data.shape

    # Generate interpolated field
    x, y = np.array(range(cols)),np.array(range(rows))
    xv, yv = np.meshgrid(x,y)
    ip = RectBivariateSpline(y, x, data, kx=1, ky=1)

    # Generate new values of interest
    xr = np.linspace(0,cols,shape[1])
    yr = np.linspace(0,rows,shape[0])

    result = ip(yr,xr,grid=True)

    return result


def computeGradient(image):
	
    kernelx = np.matrix([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    xvals = convolution(image,kernelx) 
    
    kernely = np.matrix([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    yvals = convolution(image,kernely) 
    
    #result = np.sqrt(xvals**2 + yvals**2)
	
    return xvals, yvals


def convolution(image,kernel):
    
    rows,cols = image.shape
    height, width = kernel.shape
	
    image = image.astype(np.float32)
    result = np.zeros((rows,cols)).astype(np.float32)
    
	#Determine the number of edge pixels
    re = int(height/2)
    ce = int(width/2)
	
    for i in range(0,height):
        for j in range(0,width):
            multiplier = kernel[i,j]
            result[max(0, i-re):rows-re+i,max(0, j-ce):cols-ce+j] += multiplier*image[max(0, re-i):rows+re-i,max(0, ce-j):cols+ce-j]
			
    return result
	
	
def calibrateDisp(disp, left, right, j=1, split = .95, num_iters = 500, sd1 = 0.5, sd2 = 2.0):
    
    rows, cols, ch = right.shape
    
    # Generate interpolated field
    x, y = np.array(range(cols)),np.array(range(rows))
    ipR = RectBivariateSpline(y, x, right[:,:,0], kx=1, ky=1)
    ipG = RectBivariateSpline(y, x, right[:,:,1], kx=1, ky=1)
    ipB = RectBivariateSpline(y, x, right[:,:,2], kx=1, ky=1)   
    
    xv, yv = np.meshgrid(x,y)
    trials = {}
    k = 1.0
    pivot = 0.0
	
    for n in trange(num_iters):
                
        d = j*(k*disp - pivot)
        eR = ipR(yv,xv-d,grid=False)-left[:,:,0]
        eG = ipG(yv,xv-d,grid=False)-left[:,:,1]
        eB = ipB(yv,xv-d,grid=False)-left[:,:,2]
            
        errors = (np.multiply(eR,eR) + np.multiply(eG,eG) + np.multiply(eB,eB)).flatten()

        errors.sort()
        
        # Only take elements in the first 95%
        errors = errors[:int(split*len(errors))]
        
        # Keep the total error
        trials[k]= (sum(errors)), pivot
        
        # Randomize k about the assumed depth
        k = np.random.lognormal(0,sd1)
		
		# Randomize bias as negative value
        #pivot = np.random.lognormal(0,sd2)
        pivot = np.random.normal(0.0,sd2)
		
        #print(k,bias)
		
        
    best_k = 1.0
    best_pivot = 0.0
    best_error = 1000000
    
    for k, (error, pivot) in trials.items():
        if error < best_error:
            best_k = k
            best_pivot = pivot
            best_error = error
			
    print("k: ",best_k," pivot: ",best_pivot)
            
    return best_k, best_pivot
	
def calibrateDisp2(disp, left, right, i, j, search_k=1.0, pivot= 1.0, split = .95, num_iters = 200, sd1 = 0.5):
    
    rows, cols, ch = right.shape
    
    # Generate interpolated field
    x, y = np.array(range(cols)),np.array(range(rows))
    ipR = RectBivariateSpline(y, x, right[:,:,0], kx=1, ky=1)
    ipG = RectBivariateSpline(y, x, right[:,:,1], kx=1, ky=1)
    ipB = RectBivariateSpline(y, x, right[:,:,2], kx=1, ky=1)   
    
    xv, yv = np.meshgrid(x,y)
    trials = {}
    k = search_k
	
    for n in trange(num_iters):
                
        dx = j*(k*disp - pivot)
        dy = i*(k*disp - pivot)
        eR = ipR(yv-dy,xv-dx,grid=False)-left[:,:,0]
        eG = ipG(yv-dy,xv-dx,grid=False)-left[:,:,1]
        eB = ipB(yv-dy,xv-dx,grid=False)-left[:,:,2]
            
        errors = (np.multiply(eR,eR) + np.multiply(eG,eG) + np.multiply(eB,eB)).flatten()

        errors.sort()
        
        # Only take elements in the first 95%
        errors = errors[:int(split*len(errors))]
        
        # Keep the total error
        trials[k]= sum(errors)
        
        # Randomize k about the assumed depth
        k = np.random.normal(search_k,sd1)
		
		# Randomize bias as negative value
        #pivot = np.random.lognormal(0,sd2)
		
        #print(k,bias)
		
        
    best_k = 1.0
    best_error = 1000000
    
    for k, error in trials.items():
        if error < best_error:
            best_k = k
            best_error = error
			
    print("k: ",best_k)
            
    return best_k
	
	
def preprocess(loaddir, savedir, name, modeldir, num_views=9, crop=5, epsilon=1.4, view_j=1, tuned_k=False, vectorized=True):
	
	# Attempt to find filetype
	if os.path.exists(loaddir + name + "_eslf.png"):
		print("File Type: ESLF")
		filetype = "eslf"
	elif os.path.exists(loaddir + name + ".mat"):
		print("File Type: MAT")
		filetype = "MAT"
	elif os.path.exists(loaddir + name + ".eslf.png"):
		print("File Type: New ESLF")
		filetype = "New ESLF"
	else:
		print("Light Field file type not found, assuming Synthetic Lightfield")
		filetype = "Synthetic"

	if filetype == "eslf":
		filename = loaddir + name + "_eslf.png"
		meta = loaddir + name + ".json"
		lightfield = loadESLF(filename,meta)
		depth_name = loaddir + name+ "_warp_depth.png"
		depth = loadDepth(depth_name)

		disp = 1/depth		

	elif filetype == "MAT":
		filename = loaddir + name + ".mat"
		lf = loadMAT(filename)
		depth_name = loaddir + name + ".png"
		depth = loadDepth(depth_name)


		im_r,im_c,_,_,_ = lf.shape
		rows,cols = depth.shape

		lightfield = np.zeros((im_r,im_c,rows,cols,3))

		for r in range(im_r):
			for c in range(im_c):
				lightfield[r,c] = imresize(lf[r,c,:,:,:3],(rows,cols))/255.0

		disp = 1/depth

	elif filetype == "New ESLF":
		filename = loaddir + name + ".eslf.png"
		meta = loaddir + name + ".json"
		lightfield = loadESLF(filename,meta,version=2)
		depth_name = loaddir + name+ ".depth.png"
		depth = loadDepth(depth_name)

		disp = 1/depth

	elif filetype == "Synthetic":
		print("You will need to presave the lightfield using Python 2.7")
		lightfield = np.load(savedir+"lightfield.npy")
		disp = np.load(savedir+"disp.npy")


	# Take only the given cross section
	middle = lightfield.shape[0]//2
	lightfield = lightfield[-(num_views//2) + middle:num_views//2+1 + middle,-(num_views//2) + middle:num_views//2+1 + middle]

	# Crop edge pixels if needed
	if crop > 0:
		lightfield = lightfield[:,:,crop:-crop,crop:-crop]
		disp = disp[crop:-crop,crop:-crop]

	ensure_dir(savedir[:-1]+"Gantry/")
	saveGantry(lightfield,savedir[:-1]+"Gantry/")

	stack, foci = getFocalStack(lightfield, near=0.6, far=1.6, near_step = 0.1, far_step=.15)
	ensure_dir(savedir[:-1]+"Refocused/")
	saveFocalStack(stack,foci,savedir[:-1]+"Refocused/")

	#ensure_dir(savedir[:-1]+"Visualized/")
	#saveScrollMP4(lightfield, savedir[:-1]+"Visualized/"+"LF.mp4")	

	center = getView(lightfield,0,0)
	right = getView(lightfield,0,view_j)

	# Disparity Calibration
	print("Pre-calibration:")
	print("Disparity", np.amin(disp),np.amax(disp))
	k,pivot = calibrateDisp(disp,center,right,view_j)
	print("Post-calibration:")
	print("Disparity", np.amin(k*disp - pivot), np.amax(k*disp - pivot))

	#Save the center image and disparity
	imsave(savedir+name+"_0_0_image.png",(255*center).astype(np.uint8))
	np.save(savedir+name+"_0_0_dx",k*(disp-pivot))
	np.save(savedir+name+"_0_0_dy",k*(disp-pivot))

	if tuned_k:
		best_k = k
		
		#Save fine-tuned k-values
		k_values = np.zeros((num_views,num_views))
		k_values[num_views//2, num_views//2] = best_k

	for i in trange(-(num_views//2),num_views//2+1):
		for j in range(-(num_views//2),num_views//2+1):
			if i == 0 and j == 0:
				continue

			current = getView(lightfield,i,j)

			rows, cols, ch = current.shape

			# Generate interpolated field
			x, y = np.array(range(cols)),np.array(range(rows))
			xv, yv = np.meshgrid(x,y)
			ipR = RectBivariateSpline(y, x, center[:,:,0], kx=1, ky=1)
			ipG = RectBivariateSpline(y, x, center[:,:,1], kx=1, ky=1)
			ipB = RectBivariateSpline(y, x, center[:,:,2], kx=1, ky=1)

			if tuned_k:
				k = calibrateDisp2(disp, center, current, i, j, best_k, pivot)
				k_values[j+num_views//2,i+num_views//2] = k

			disp_x = j*(k*disp - pivot)
			disp_y = i*(k*disp - pivot)

			dx_max = int(np.ceil(np.amax(np.abs(disp_x))))*np.sign(j)
			dy_max = int(np.ceil(np.amax(np.abs(disp_y))))*np.sign(i)

			#print(dx_max, dy_max)

			xmap = xv-disp_x
			ymap = yv-disp_y

			dx = np.zeros((rows,cols))
			dy = np.zeros((rows,cols))       

			mask = np.zeros((rows,cols))

			if vectorized:
			
				occs = np.zeros((rows,cols)) 
			
				if j == 0:
					xshifts = [0]
				else:
					xshifts = range(-abs(dx_max),abs(dx_max) + 1)

				if i == 0:
					yshifts = [0]
				else:
					yshifts = range(-abs(dy_max),abs(dy_max) + 1)
					
				for sx in xshifts:
					for sy in yshifts:
					
						if sx < 0:
							x_vals = range(0,cols+sx)
							dx_vals = range(-sx,cols)
							offset_x = -sx
						elif sx == 0:
							x_vals = range(0,cols)
							dx_vals = range(0,cols)
							offset_x = 0
						else:
							x_vals = range(sx,cols)
							dx_vals = range(0,cols-sx)
							offset_x = 0
							
						if sy < 0:
							y_vals = range(0,rows+sy)
							dy_vals = range(-sy,rows)
							offset_y = -sy
						elif sy == 0:
							y_vals = range(0,rows)
							dy_vals = range(0,rows)
							offset_y = 0
						else:
							y_vals = range(sy,rows)
							dy_vals = range(0,rows-sy)
							offset_y = 0
				
						
						window_x = xmap[:,x_vals][y_vals,:] - xv[:,dx_vals][dy_vals,:]
						window_dx = disp_x[:,x_vals][y_vals,:]
						window_y = ymap[:,x_vals][y_vals,:] - yv[:,dx_vals][dy_vals,:]
						window_dy = disp_y[:,x_vals][y_vals,:]

						valid = (np.square(window_x)+np.square(window_y))<epsilon
						window_dx *= valid
						window_dy *= valid
						movement = np.square(window_dx) + np.square(window_dy)
						
						indices = np.where((movement>occs[:,x_vals][y_vals,:]))
						occs[indices[0][:] + offset_y, indices[1][:] + offset_x] = movement[indices]
						dx[indices[0][:] + offset_y, indices[1][:] + offset_x] = window_dx[indices]
						dy[indices[0][:] + offset_y, indices[1][:] + offset_x] = window_dy[indices]
						mask[indices[0][:] + offset_y, indices[1][:] + offset_x] = 1
				
			
			else:

				# Make new view disparity map
				for xR in x:
					for yR in y:

						if j < 0:
							xvals = range(max(xR+dx_max,0),min(xR-dx_max,cols))
						if j == 0:
							xvals = [xR]
						if j > 0:
							xvals = range(max(xR-dx_max,0),min(xR+dx_max,cols))

						if i < 0:
							yvals = range(max(yR+dy_max,0),min(yR-dy_max,rows))
						if i == 0:
							yvals = [yR]
						if i > 0:
							yvals = range(max(yR-dy_max,0),min(yR+dy_max,rows))

						window_x = xmap[:,xvals][yvals,:] - xR
						window_dx = disp_x[:,xvals][yvals,:]
						window_y = ymap[:,xvals][yvals,:] - yR
						window_dy = disp_y[:,xvals][yvals,:]

						indices = np.where((np.square(window_x)+np.square(window_y)<epsilon))

						# Account for 0 to 1 mappings and many to one mappings
						if indices[0].size:
							# Look for Maximum Disparity
							possibles = np.square(window_dx[indices]) + np.square(window_dy[indices])
							index = np.argmax(possibles)
							# or....... 
							# Look for Closest Landing Pixel
							#possibles = np.square(window_x[indices]) + np.square(window_y[indices])
							#index = np.argmin(possibles)
							
							dx[yR,xR] = window_dx[indices][index]
							dy[yR,xR] = window_dy[indices][index]
							mask[yR,xR] = 1
						else:
							mask[yR,xR] = 0

			mask *= calcSimilarity(xv,yv,dx,dy,ipR,ipG,ipB,current)

			# Post Process mask
			#mask = processMask(mask, 2, 1, 2)
			#plt.imshow(mask,cmap="Greys_r",vmin=0,vmax=1);plt.show()

			# Save outputs
			imsave(savedir + name + "_" +str(i) +"_" + str(j)+"_image.png",(255*current).astype(np.uint8))
			imsave(savedir + name + "_" +str(i) +"_" + str(j)+"_mask.png",(255*mask).astype(np.uint8))
			#imsave(savedir + name + "_" +str(i) +"_" + str(j)+"_dx.png",dx)
			#imsave(savedir + name + "_" +str(i) +"_" + str(j)+"_dy.png",dy)
			np.save(savedir + name + "_" +str(i) +"_" + str(j)+"_mask",mask)
			np.save(savedir + name + "_" +str(i) +"_" + str(j)+"_dx",dx)
			np.save(savedir + name + "_" +str(i) +"_" + str(j)+"_dy",dy)

	# Make warped versions of images
	lf = loadViewsNumbered(savedir, num_views, num_views, desc="_image")

	center = getView(lf,0,0)
	img_shape = center.shape[:2]

	masks = np.ones((num_views,num_views,img_shape[0],img_shape[1])).astype(np.float32)
	warped_centers = np.zeros((num_views,num_views,img_shape[0],img_shape[1],3)).astype(np.float32)

	for i in range(-(num_views//2),(num_views//2)+1):
		for j in range(-(num_views//2),(num_views//2)+1):
			if i == 0 and j == 0:
				warped_centers[i,j] = center
				continue

			ic = i + num_views//2
			jc = j + num_views//2

			# Warp the center based on the disparity maps
			dx = np.load(savedir + name + "_" + str(i) + "_" + str(j) + "_dx.npy")
			dy = np.load(savedir + name + "_" + str(i) + "_" + str(j) + "_dy.npy")
			mask = np.load(savedir + name + "_" + str(i) + "_" + str(j) + "_mask.npy")

			dx = resizeNP(dx,img_shape).astype(np.float32)
			dy = resizeNP(dy,img_shape).astype(np.float32)
			mask = resizeNP(mask,img_shape).astype(np.float32)

			warped_center = warpViewNP(center, dx, dy)
			masks[ic,jc] = mask
			warped_centers[ic,jc] = warped_center

	ensure_dir(savedir[:-1]+"WarpVisualized/")
	
	if tuned_k:
		np.save(savedir[:-1]+"WarpVisualized/k_values",k_values)
	
	saveScrollMP4(lf, savedir[:-1]+"WarpVisualized/lf.mp4")
	#saveScrollMP4(warped_centers, savedir[:-1]+"WarpVisualized/warped_no_mask.mp4")
	saveScrollMP4(np.expand_dims(np.maximum(0,masks),axis=4)*warped_centers, savedir[:-1]+"WarpVisualized/warped.mp4")
	#saveScrollMP4(.5*warped_centers + .5*lf[:,:,:,:,:3], savedir[:-1]+"WarpVisualized/blended.mp4")
	saveScrollMP4(np.expand_dims(np.maximum(0,masks),axis=4)*warped_centers + (1-np.expand_dims(np.maximum(0,masks),axis=4))*lf[:,:,:,:,:3], savedir[:-1]+"WarpVisualized/maskblended.mp4")
	#temp = np.expand_dims(np.maximum(0,masks),axis=4)
	#saveScrollMP4(np.concatenate((temp,temp,temp),axis=4), savedir[:-1]+"WarpVisualized/masks.mp4")

	print("Done Processing Light Field!")