import numpy as np
from tqdm import tqdm
from scipy.misc import imresize, imsave

def saveImage(im, filename):
	imsave(filename,im)

def saveGantry(lf, directory):
	count=0
	for r in lf:
		for c in r:
			saveImage(c,directory+"{:3d}".format(count)+".png")
			count+=1
	
def saveFocalStack(stack, foci, filename, filetype=".png"):
	
	for i in range(len(stack)):
		imsave(filename + "_" + "%.2f" % foci[i] + "_" + filetype, stack[i])
		
def saveEpipolar(lf, filename="output.png", crop=None, center_fn = "center.png"):
	h,w,rows,cols,ch = lf.shape

	epipolar = np.zeros((w*rows,h*cols,ch))

	for v in range(h):
		for y in range(rows):
			for u in range(w):
				epipolar[y*w+u,v*cols:(v+1)*cols] = lf[v,u,y,:]
			
	if crop:
		center_row = w*rows//2
		center_col = h*cols//2
		
		r = crop[0]*w
		c = crop[1]
		
		epipolar = epipolar[center_row - r//2:center_row+r//2+1,center_col - c//2:center_col+c//2+1]
		
		# Mark the grabbed region on the center image
		center_image = lf[h//2,w//2]
		center_image[rows//2 - crop[0]//2-1:rows//2 - crop[0]//2-1+2, cols//2 - crop[1]//2 - 1:cols//2 + crop[1]//2+1 + 1] = 1
		center_image[rows//2 + crop[0]//2-1:rows//2 + crop[0]//2-1+2, cols//2 - crop[1]//2 - 1:cols//2 + crop[1]//2+1 + 1] = 1
		center_image[rows//2 - crop[0]//2 - 1:rows//2 + crop[0]//2+ 1, cols//2 - crop[1]//2 - 1: cols//2 - crop[1]//2 + 2 ] = 1
		center_image[rows//2 - crop[0]//2 - 1:rows//2 + crop[0]//2+ 1, cols//2 + crop[1]//2 - 1: cols//2 + crop[1]//2 + 2] = 1
		
		imsave(center_fn,center_image)
			
	imsave(filename,epipolar)
		
def saveGIF(images, filename="output.gif"):
	import imageio
	imageio.mimsave(filename, images, fps=5)

# Note: To use this function,  you may need to install the ffmpeg codec to your computer.
def saveMP4(images, filename="output.mp4", my_fps=15):
	print("Saving...")
		
	import matplotlib.animation as animation
	import matplotlib.pyplot as plt
	mp4Writer = animation.writers['ffmpeg']
	the_writer = mp4Writer(fps=my_fps, metadata=dict(artist='Me'))
	fig = plt.figure()
	ims = []
	
	rows, cols, _ = images[0].shape
		
	fig.set_size_inches(20,20*rows/cols)
	
	#plt.axis([0, cols, rows, 0])
	plt.axis('off')
	
	for image in images:
		ims.append([plt.imshow(image,vmin=0,vmax=255,animated=True)])
		
	im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=True)
	im_ani.save(filename, writer=the_writer)
	plt.close(fig)
	print("Done")
	
def saveScrollMP4(lf, filename="output.mp4", bgcolor = [0,1,1], fgcolor = [1,0,0], marker_size=10):
	
	h, w, im_h, _, _ = lf.shape
	
	images = []
	
	# Horizontal Scroll 
	switch = -1
	for i in range(h):
		switch *= -1
		for j in range(w)[::switch]:
			im = lf[i,j,:,:,:3]
			# Make viewer bar
			position = np.zeros((im_h,(w+2)*marker_size,3))
			position[marker_size:(h+1)*marker_size,marker_size:(w+1)*marker_size] = bgcolor
			position[(i+1)*marker_size:(i+2)*marker_size,(j+1)*marker_size:(j+2)*marker_size] = fgcolor

			result = np.concatenate((position, im), axis=1)
			
			images.append(result)
			
	# Vertical Scroll 
	switch = 1
	for j in range(w)[::-1]:
		switch *= -1
		for i in range(h)[::switch]:
			im = lf[i,j,:,:,:3]
			# Make viewer bar
			position = np.zeros((im_h,(w+2)*marker_size,3)).astype(np.float32)
			position[marker_size:(h+1)*marker_size,marker_size:(w+1)*marker_size] = bgcolor
			position[(i+1)*marker_size:(i+2)*marker_size,(j+1)*marker_size:(j+2)*marker_size] = fgcolor

			result = np.concatenate((position, im),axis=1)
			
			images.append(result)
			
	saveMP4(images,filename)
	
def saveNPY(lf, filename):
	return np.save(filename, lf)