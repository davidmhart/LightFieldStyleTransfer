from StyleNet import utils
from StyleNet import transformer_net as tn
from StyleNet import calculate_loss as cl

from LoadLightField import loadViewsNumbered
from LightFieldFunctions import getFocalStack
from SaveLightField import saveGantry, saveFocalStack, saveScrollMP4

import torch
from torch.optim import Adam
from torch.autograd import Variable
from tqdm import trange, tqdm
import os
from time import time

import numpy as np

class GramMatrix(torch.nn.Module):
    def forward(self, input):
        b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(b * c * d)
		
def encode(image, model):

	# Based on transformer net
	y = model.relu(model.in1(model.conv1(image)))
	y = model.relu(model.in2(model.conv2(y)))
	y = model.relu(model.in3(model.conv3(y)))
	y = model.res1(y)
	y = model.res2(y)
	y = model.res3(y)
	return y
	
def decode(features, model):

	y = model.res4(features)
	y = model.res5(y)
	y = model.relu(model.in4(model.deconv1(y)))
	y = model.relu(model.in5(model.deconv2(y)))
	y = model.deconv3(y)
	return y
	
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

def averageBlur(image):
    
    kernel = np.matrix([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]])
    
    return (convolution(image,kernel)/9)

def edgeDetect(image):
    

    kernelx = np.matrix([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    xvals = convolution(image,kernelx) 
    
    kernely = np.matrix([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    yvals = convolution(image,kernely) 
    
    result = np.sqrt(xvals**2 + yvals**2)

    return result
	
def gibbsEnergy(image):
    
    gibbs = torch.zeros(image.shape[1:])
	
    if torch.cuda.is_available():
        gibbs = gibbs.cuda()
    
    # 8-connected
    gibbs[1:-1,1:-1] += (image[0,1:-1,1:-1] - image[0,0:-2,0:-2])**2 + (image[1,1:-1,1:-1] - image[1,0:-2,0:-2])**2 + (image[2,1:-1,1:-1] - image[2,0:-2,0:-2])**2
    gibbs[1:-1,1:-1] += (image[0,1:-1,1:-1] - image[0,0:-2,1:-1])**2 + (image[1,1:-1,1:-1] - image[1,0:-2,1:-1])**2 + (image[2,1:-1,1:-1] - image[2,0:-2,1:-1])**2
    gibbs[1:-1,1:-1] += (image[0,1:-1,1:-1] - image[0,0:-2,2:])**2 + (image[1,1:-1,1:-1] - image[1,0:-2,2:])**2 + (image[2,1:-1,1:-1] - image[2,0:-2,2:])**2
    gibbs[1:-1,1:-1] += (image[0,1:-1,1:-1] - image[0,1:-1,0:-2])**2 + (image[1,1:-1,1:-1] - image[1,1:-1,0:-2])**2 + (image[2,1:-1,1:-1] - image[2,1:-1,0:-2])**2
    gibbs[1:-1,1:-1] += (image[0,1:-1,1:-1] - image[0,1:-1,1:-1])**2 + (image[1,1:-1,1:-1] - image[1,1:-1,1:-1])**2 + (image[2,1:-1,1:-1] - image[2,1:-1,1:-1])**2
    gibbs[1:-1,1:-1] += (image[0,1:-1,1:-1] - image[0,1:-1,2:])**2 + (image[1,1:-1,1:-1] - image[1,1:-1,2:])**2 + (image[2,1:-1,1:-1] - image[2,1:-1,2:])**2
    gibbs[1:-1,1:-1] += (image[0,1:-1,1:-1] - image[0,2:,0:-2])**2 + (image[1,1:-1,1:-1] - image[1,2:,0:-2])**2 + (image[2,1:-1,1:-1] - image[2,2:,0:-2])**2
    gibbs[1:-1,1:-1] += (image[0,1:-1,1:-1] - image[0,2:,1:-1])**2 + (image[1,1:-1,1:-1] - image[1,2:,1:-1])**2 + (image[2,1:-1,1:-1] - image[2,2:,1:-1])**2
    gibbs[1:-1,1:-1] += (image[0,1:-1,1:-1] - image[0,2:,2:])**2 + (image[1,1:-1,1:-1] - image[1,2:,2:])**2 + (image[2,1:-1,1:-1] - image[2,2:,2:])**2
    
    return gibbs

		

def LFStyleTransferBP(name, LFdir, loaddir, savedir, num_views, fuse_features=True, fuse_images=False, perceptual_loss=False, gibbs_loss=False, analysis_only=False, learning_rate = 1e-2, epochs = 50, beta = 5, gamma = 500, kappa = 1000, convergence_num=5):

	utils.ensure_dir(savedir)
	count = 0
	modeldir = loaddir + "models/"
	model_fns = os.listdir(modeldir)
	
	# Load for computing loss
	vggdir = loaddir+"vgg16/"
	styledir = loaddir+"styles/"
	vgg = utils.Vgg16()
	vgg.load_state_dict(torch.load(vggdir+"vgg16.weight"))
	if torch.cuda.is_available():
		vgg.cuda()
	
	for fn in model_fns:
	
		currentdir = savedir+"Model"+str(count)+"/"
		utils.ensure_dir(currentdir)
	
		start = time()
	
		if not analysis_only:
			style_model = tn.TransformerNet()
			style_model.load_state_dict(torch.load(modeldir+fn))
			
			if torch.cuda.is_available():
				style_model.cuda()

			# Once for the center
			content_image = utils.tensor_load_rgbimage(LFdir+name+"_0_0_image.png")#,shape=img_shape)
			content_image = content_image.unsqueeze(0)

			if torch.cuda.is_available():
				content_image = content_image.cuda()
			content_image = Variable(utils.preprocess_batch(content_image))
					
			center_features = encode(content_image,style_model)
			center = decode(center_features,style_model)
			utils.tensor_save_rgbimage(center.data[0], currentdir+name+"_0_0.png", torch.cuda.is_available())
			
			img_shape = center.size()[2:]
			features_shape = center_features.size()[2:]
			
			mse_loss = torch.nn.MSELoss()
			style_fn = os.listdir(styledir)[count]
			gram_style = cl.get_gram(styledir+style_fn,vgg)
			
			# Once for every other image
			switch = -1
			for i in trange(-(num_views//2),(num_views//2)+1):
				switch *= -1
				jvals = range(-(num_views//2),(num_views//2)+1)[::switch]
				for j in jvals:
						
					if i == -(num_views//2) and j== -(num_views//2):
						epochs_ = 2*epochs
					elif i == -(num_views//2) and j== -(num_views//2)+1:
						epochs_ = int(1.75*epochs)
					elif i == -(num_views//2) and j== -(num_views//2)+2:
						epochs_ = int(1.5*epochs)
					else:
						epochs_ = epochs
						
					if i == 0 and j == 0:
						continue
					
					content_image = utils.tensor_load_rgbimage(LFdir+name+"_"+str(i)+"_"+str(j)+"_image.png",shape=img_shape)
					content_image = content_image.unsqueeze(0)

					if torch.cuda.is_available():
						content_image = content_image.cuda()
					content_image = Variable(utils.preprocess_batch(content_image),requires_grad=True)
					
					# Warp the center based on the disparity maps
					dx = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_dx.npy")
					dy = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_dy.npy")
					mask = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_mask.npy")
					
					dx_features = utils.resizeNP(dx,features_shape).astype(np.float32)/4 #Scale for mapping
					dy_features = utils.resizeNP(dy,features_shape).astype(np.float32)/4
					mask_features = utils.resizeNP(mask,features_shape).astype(np.float32)
					
					warped_center_features = utils.warpFeatures(center_features, dx_features, dy_features)
					if torch.cuda.is_available():
						warped_center_features = warped_center_features.cuda()
						
					mask_features = Variable(torch.from_numpy(mask_features),requires_grad=False)
					if torch.cuda.is_available():
						mask_features = mask_features.cuda()
						
					dx = utils.resizeNP(dx,img_shape).astype(np.float32)
					dy = utils.resizeNP(dy,img_shape).astype(np.float32)
					mask = utils.resizeNP(mask,img_shape).astype(np.float32)
					
					if gibbs_loss:
						grad_mask = edgeDetect(mask)
						grad_mask = Variable(torch.from_numpy(grad_mask),requires_grad=False)
						if torch.cuda.is_available():
							grad_mask = grad_mask.cuda()
					
					warped_center = utils.warpView(center, dx, dy)
					if torch.cuda.is_available():
						warped_center = warped_center.cuda()
						
					mask = Variable(torch.from_numpy(mask),requires_grad=False)
					if torch.cuda.is_available():
						mask = mask.cuda()

					xc = Variable(content_image.data.clone())
					xc = utils.subtract_imagenet_mean_batch(xc)
					features_xc = vgg(xc)
					f_xc_c = Variable(features_xc[1].data, requires_grad=False)
						
					style_model.train()
					
					
					# Setup deep learning parameters
					params = style_model.parameters()
					optimizer = torch.optim.Adam(params, lr=learning_rate)
					start_num = 25
					prev_loss = 0
					current_loss = 0
					stable_count = 0
					loss_list = []
					
					for epoch_num in range(epochs_): 
						
						if fuse_features:
							# Encode the view, compare with center features, and then decode it
							view_features = encode(content_image,style_model)
							view_features = mask_features*(warped_center_features) + (1-mask_features)*view_features
							output = decode(view_features,style_model)
						else:
							output = style_model(content_image)
						
						if fuse_images:
							output = mask*warped_center + (1-mask)*output
						
						y = output
						
						# Remove useless pixels and calculate loss
						pixel_loss = mask*((output[0,0]-warped_center[0,0])**2 + (output[0,1]-warped_center[0,1])**2 + (output[0,2]-warped_center[0,2])**2)
						disparity_loss = torch.sum(pixel_loss)
						total_loss = gamma* disparity_loss
						
						if epoch_num == start_num:
							prev_loss = disparity_loss
							current_loss = disparity_loss
							loss_list.append(current_loss.item())
						elif epoch_num > start_num:
							prev_loss = current_loss
							current_loss = disparity_loss
							loss_list.append(current_loss.item())
						
							
						if perceptual_loss:
							y = utils.subtract_imagenet_mean_batch(y)

							features_y = vgg(y)
							content_loss = mse_loss(features_y[1], f_xc_c)

							style_loss = 0.
							for m in range(len(features_y)):
								gram_s = Variable(gram_style[m].data, requires_grad=False)
								gram_y = utils.gram_matrix(features_y[m])
								style_loss += mse_loss(gram_y, gram_s[:1, :, :])
							
							total_loss += content_loss + beta*style_loss
						
						if gibbs_loss:
							energy = grad_mask*gibbsEnergy(output[0])
							total_loss += kappa*torch.sum(energy)
							
						#print(total_loss.item())
							
						# Training step
						optimizer.zero_grad()
						total_loss.backward()
						optimizer.step()
					
						#Break if convergence is reached
						if epoch_num > start_num and current_loss > prev_loss and convergence_num:
							stable_count += 1
							if stable_count >= convergence_num:
								print("View",i,",",j,"converged after",epoch_num,"iterations")
								break
					
					# Save output
					style_model.eval()
					if fuse_features:
						# Encode the view, compare with center features, and then decode it
						view_features = encode(content_image,style_model)
						view_features = mask_features*(warped_center_features) + (1-mask_features)*view_features
						output = decode(view_features,style_model)
					else:
						output = style_model(content_image)
					
					if fuse_images:
						output = mask*warped_center + (1-mask)*output
					
					#import matplotlib.pyplot as plt
					#plt.plot(loss_list)
					#plt.show()
					
					utils.tensor_save_rgbimage(output.data[0], currentdir+name+"_"+str(i)+"_"+str(j)+".png", torch.cuda.is_available())

		end = time()
					
		lf = loadViewsNumbered(currentdir, num_views, num_views)
		utils.ensure_dir(currentdir[:-1]+"Gantry/")
		saveGantry(lf,currentdir[:-1]+"Gantry/")
		
		stack, foci = getFocalStack(lf, near=0.6, far=1.6, near_step = 0.1, far_step=.15)
		utils.ensure_dir(currentdir[:-1]+"Refocused/")
		saveFocalStack(stack,foci,currentdir[:-1]+"Refocused/")
		
		utils.ensure_dir(currentdir[:-1]+"Visualized/")
		saveScrollMP4(lf, currentdir[:-1]+"Visualized/"+"Model"+str(count)+".mp4")

		lf_original = loadViewsNumbered(LFdir, num_views, num_views, desc="_image")
		style_fn = styledir+os.listdir(styledir)[count]
		loss_matrix = cl.compute_LF_loss(lf_original,lf,vgg,style_fn)
		disp_matrix = cl.compute_LF_disp_loss(lf,name,LFdir)
		cl.save_loss_matrix(loss_matrix,disp_matrix,currentdir[:-1]+"Loss/")
		
		endstats = time()
		
		count += 1
		
	if not analysis_only:
		file = open(savedir+"Time.txt","w") 
		file.write("Model Time: " + str(end-start) + "\n")
		file.write("Stats Time: " + str(endstats-end) + "\n")
		file.write("Total Time (per model): " + str(endstats-start) + "\n")
		file.write("Total Time (all models): " + str((endstats-start)*(count+1)) + "\n")
		file.close() 
		
		
def LFStyleTransferPostOpt(name, LFdir, loaddir, savedir, num_views, fuse_features=True, fuse_images=False, perceptual_loss=False, gibbs_loss=False, analysis_only=False, learning_rate = 1e0, epochs = 500, beta = 5, gamma = 500, kappa = 1000, convergence_num = 20):

	utils.ensure_dir(savedir)
	count = 0
	modeldir = loaddir + "models/"
	model_fns = os.listdir(modeldir)
	
	# Load for computing loss
	vggdir = loaddir+"vgg16/"
	styledir = loaddir+"styles/"
	vgg = utils.Vgg16()
	vgg.load_state_dict(torch.load(vggdir+"vgg16.weight"))
	if torch.cuda.is_available():
		vgg.cuda()
	
	for fn in model_fns:
	
		currentdir = savedir+"Model"+str(count)+"/"
		utils.ensure_dir(currentdir)
	
		start = time()
	
		if not analysis_only:
			style_model = tn.TransformerNet()
			style_model.load_state_dict(torch.load(modeldir+fn))
			
			if torch.cuda.is_available():
				style_model.cuda()

			# Once for the center
			content_image = utils.tensor_load_rgbimage(LFdir+name+"_0_0_image.png")#,shape=img_shape)
			content_image = content_image.unsqueeze(0)

			if torch.cuda.is_available():
				content_image = content_image.cuda()
			content_image = Variable(utils.preprocess_batch(content_image))
					
			center_features = encode(content_image,style_model)
			center = decode(center_features,style_model)
			utils.tensor_save_rgbimage(center.data[0], currentdir+name+"_0_0.png", torch.cuda.is_available())
			
			img_shape = center.size()[2:]
			features_shape = center_features.size()[2:]
			
			mse_loss = torch.nn.MSELoss()
			style_fn = os.listdir(styledir)[count]
			gram_style = cl.get_gram(styledir+style_fn,vgg)
			
			# Once for every other image
			for i in trange(-(num_views//2),(num_views//2)+1):
				for j in range(-(num_views//2),(num_views//2)+1):
						
					if i == 0 and j == 0:
						continue
					
					content_image = utils.tensor_load_rgbimage(LFdir+name+"_"+str(i)+"_"+str(j)+"_image.png",shape=img_shape)
					content_image = content_image.unsqueeze(0)

					if torch.cuda.is_available():
						content_image = content_image.cuda()
					content_image = Variable(utils.preprocess_batch(content_image),requires_grad=True)
					
					# Warp the center based on the disparity maps
					dx = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_dx.npy")
					dy = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_dy.npy")
					mask = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_mask.npy")
					
					dx_features = utils.resizeNP(dx,features_shape).astype(np.float32)/4 #Scale for mapping
					dy_features = utils.resizeNP(dy,features_shape).astype(np.float32)/4
					mask_features = utils.resizeNP(mask,features_shape).astype(np.float32)
					
					warped_center_features = utils.warpFeatures(center_features, dx_features, dy_features)
					if torch.cuda.is_available():
						warped_center_features = warped_center_features.cuda()
						
					mask_features = Variable(torch.from_numpy(mask_features),requires_grad=False)
					if torch.cuda.is_available():
						mask_features = mask_features.cuda()
						
					dx = utils.resizeNP(dx,img_shape).astype(np.float32)
					dy = utils.resizeNP(dy,img_shape).astype(np.float32)
					mask = utils.resizeNP(mask,img_shape).astype(np.float32)
					
					if gibbs_loss:
						grad_mask = edgeDetect(mask)
						grad_mask = Variable(torch.from_numpy(grad_mask),requires_grad=False)
						if torch.cuda.is_available():
							grad_mask = grad_mask.cuda()
					
					warped_center = utils.warpView(center, dx, dy)
					if torch.cuda.is_available():
						warped_center = warped_center.cuda()
						
					mask = Variable(torch.from_numpy(mask),requires_grad=False)
					if torch.cuda.is_available():
						mask = mask.cuda()

					xc = Variable(content_image.data.clone())
					xc = utils.subtract_imagenet_mean_batch(xc)
					features_xc = vgg(xc)
					f_xc_c = Variable(features_xc[1].data, requires_grad=False)
						
					if fuse_features:
						# Encode the view, compare with center features, and then decode it
						view_features = encode(content_image,style_model)
						view_features = mask_features*(warped_center_features) + (1-mask_features)*view_features
						output = decode(view_features,style_model)
					else:
						output = style_model(content_image)
					
					if fuse_images:
						output = mask*warped_center + (1-mask)*output
					
					if torch.cuda.is_available():
						output = Variable(output.data.cuda(), requires_grad=True)
					else:
						output = Variable(output.data, requires_grad=True)
					
					# Setup deep learning parameters
					params = [output]
					optimizer = torch.optim.Adam(params, lr=learning_rate)
					start_num = 25
					prev_loss = 0
					current_loss = 0
					stable_count = 0
					loss_list = []
						
					style_model.train()
					for epoch_num in range(epochs): 
						
						y = output
						
						# Remove useless pixels and calculate loss
						pixel_loss = mask*((output[0,0]-warped_center[0,0])**2 + (output[0,1]-warped_center[0,1])**2 + (output[0,2]-warped_center[0,2])**2)
						disparity_loss = torch.sum(pixel_loss)
						total_loss = gamma* disparity_loss
							
						if epoch_num == start_num:
							prev_loss = disparity_loss
							current_loss = disparity_loss
							loss_list.append(current_loss.item())
						elif epoch_num > start_num:
							prev_loss = current_loss
							current_loss = disparity_loss
							loss_list.append(current_loss.item())

							
						if perceptual_loss:
							y = utils.subtract_imagenet_mean_batch(y)

							features_y = vgg(y)
							content_loss = mse_loss(features_y[1], f_xc_c)

							style_loss = 0.
							for m in range(len(features_y)):
								gram_s = Variable(gram_style[m].data, requires_grad=False)
								gram_y = utils.gram_matrix(features_y[m])
								style_loss += mse_loss(gram_y, gram_s[:1, :, :])
							
							total_loss += content_loss + beta*style_loss
													
						if gibbs_loss:
							energy = grad_mask*gibbsEnergy(output[0])
							total_loss += kappa*torch.sum(energy)
							
						# Training step
						optimizer.zero_grad()
						total_loss.backward()
						optimizer.step()
						
						#Break if convergence is reached
						if epoch_num > start_num and current_loss > prev_loss and convergence_num:
							stable_count += 1
							if stable_count >= convergence_num:
								print("View",i,",",j,"converged after",epoch_num,"iterations")
								break
					
						
					# Save output
				
					#import matplotlib.pyplot as plt
					#plt.plot(loss_list)
					#plt.show()
				
					utils.tensor_save_rgbimage(output.data[0], currentdir+name+"_"+str(i)+"_"+str(j)+".png", torch.cuda.is_available())

		end = time()
					
		lf = loadViewsNumbered(currentdir, num_views, num_views)
		utils.ensure_dir(currentdir[:-1]+"Gantry/")
		saveGantry(lf,currentdir[:-1]+"Gantry/")
		
		stack, foci = getFocalStack(lf, near=0.6, far=1.6, near_step = 0.1, far_step=.15)
		utils.ensure_dir(currentdir[:-1]+"Refocused/")
		saveFocalStack(stack,foci,currentdir[:-1]+"Refocused/")
		
		utils.ensure_dir(currentdir[:-1]+"Visualized/")
		saveScrollMP4(lf, currentdir[:-1]+"Visualized/"+"Model"+str(count)+".mp4")

		lf_original = loadViewsNumbered(LFdir, num_views, num_views, desc="_image")
		style_fn = styledir+os.listdir(styledir)[count]
		loss_matrix = cl.compute_LF_loss(lf_original,lf,vgg,style_fn)
		disp_matrix = cl.compute_LF_disp_loss(lf,name,LFdir)
		cl.save_loss_matrix(loss_matrix,disp_matrix,currentdir[:-1]+"Loss/")
		
		endstats = time()
		
		count += 1
		
	if not analysis_only:
		file = open(savedir+"Time.txt","w") 
		file.write("Model Time: " + str(end-start) + "\n")
		file.write("Stats Time: " + str(endstats-end) + "\n")
		file.write("Total Time (per model): " + str(endstats-start) + "\n")
		file.write("Total Time (all models): " + str((endstats-start)*(count+1)) + "\n")
		file.close() 

def LFStyleTransferNoOpt(name, LFdir, loaddir, savedir, num_views, fuse_features=True, fuse_images=False, analysis_only=False):

	utils.ensure_dir(savedir)
	modeldir = loaddir + "models/"
	
	count = 0
	model_fns = os.listdir(modeldir)
	
	# Load for computing loss
	vggdir = loaddir+"vgg16/"
	styledir = loaddir+"styles/"
	vgg = utils.Vgg16()
	vgg.load_state_dict(torch.load(vggdir+"vgg16.weight"))
	if torch.cuda.is_available():
		vgg.cuda()
	
	for fn in model_fns:
	
		currentdir = savedir+"Model"+str(count)+"/"
		utils.ensure_dir(currentdir)
	
		start = time()
	
		if not analysis_only:
	
			style_model = tn.TransformerNet()
			style_model.load_state_dict(torch.load(modeldir+fn))
			
			if torch.cuda.is_available():
				style_model.cuda()

			# Once for the center
			content_image = utils.tensor_load_rgbimage(LFdir+name+"_0_0_image.png")
			content_image = content_image.unsqueeze(0)

			if torch.cuda.is_available():
				content_image = content_image.cuda()
			content_image = Variable(utils.preprocess_batch(content_image))
				
			center_features = encode(content_image,style_model)
			center = decode(center_features,style_model)
			utils.tensor_save_rgbimage(center.data[0], currentdir+name+"_0_0.png", torch.cuda.is_available())
			
			img_shape = center.size()[2:]
			features_shape = center_features.size()[2:]
			
			# Once for every other image
			for i in trange(-(num_views//2),(num_views//2)+1):
				for j in range(-(num_views//2),(num_views//2)+1):
				
					if i == 0 and j == 0:
						continue
				
					content_image = utils.tensor_load_rgbimage(LFdir+name+"_"+str(i)+"_"+str(j)+"_image.png")
					content_image = content_image.unsqueeze(0)

					if torch.cuda.is_available():
						content_image = content_image.cuda()
					content_image = Variable(utils.preprocess_batch(content_image))
					
					# Warp the center based on the disparity maps
					dx = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_dx.npy")
					dy = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_dy.npy")
					mask = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_mask.npy")
					
					dx_features = utils.resizeNP(dx,features_shape).astype(np.float32)/4 #Scale for mapping
					dy_features = utils.resizeNP(dy,features_shape).astype(np.float32)/4
					mask_features = utils.resizeNP(mask,features_shape).astype(np.float32)
					
					warped_center_features = utils.warpFeatures(center_features, dx_features, dy_features)
					if torch.cuda.is_available():
						warped_center_features = warped_center_features.cuda()
						
					mask_features = Variable(torch.from_numpy(mask_features),requires_grad=False)
					if torch.cuda.is_available():
						mask_features = mask_features.cuda()
						
					dx = utils.resizeNP(dx,img_shape).astype(np.float32)
					dy = utils.resizeNP(dy,img_shape).astype(np.float32)
					mask = utils.resizeNP(mask,img_shape).astype(np.float32)
					
					warped_center = utils.warpView(center, dx, dy)
					if torch.cuda.is_available():
						warped_center = warped_center.cuda()
						
					mask = Variable(torch.from_numpy(mask),requires_grad=False)
					if torch.cuda.is_available():
						mask = mask.cuda()

					
					if fuse_features:
						# Encode the view, compare with center features, and then decode it
						view_features = encode(content_image,style_model)
						view_features = mask_features*(warped_center_features) + (1-mask_features)*view_features
						output = decode(view_features,style_model)
					else:
						output = style_model(content_image)
					
					if fuse_images:
						output = mask*warped_center + (1-mask)*output
					
					utils.tensor_save_rgbimage(output.data[0], currentdir+name+"_"+str(i)+"_"+str(j)+".png", torch.cuda.is_available())

			
		end = time()
			
		lf = loadViewsNumbered(currentdir, num_views, num_views)
		utils.ensure_dir(currentdir[:-1]+"Gantry/")
		saveGantry(lf,currentdir[:-1]+"Gantry/")
		
		stack, foci = getFocalStack(lf, near=0.6, far=1.6, near_step = 0.1, far_step=.15)
		utils.ensure_dir(currentdir[:-1]+"Refocused/")
		saveFocalStack(stack,foci,currentdir[:-1]+"Refocused/")
		
		utils.ensure_dir(currentdir[:-1]+"Visualized/")
		saveScrollMP4(lf, currentdir[:-1]+"Visualized/"+"Model"+str(count)+".mp4")

		lf_original = loadViewsNumbered(LFdir, num_views, num_views, desc="_image")
		style_fn = styledir+os.listdir(styledir)[count]
		loss_matrix = cl.compute_LF_loss(lf_original,lf,vgg,style_fn)
		disp_matrix = cl.compute_LF_disp_loss(lf,name,LFdir)
		cl.save_loss_matrix(loss_matrix,disp_matrix,currentdir[:-1]+"Loss/")
		
		#print(loss_matrix)
		endstats = time()
		
		count += 1

	if not analysis_only:
		file = open(savedir+"Time.txt","w") 
		file.write("Model Time: " + str(end-start) + "\n")
		file.write("Stats Time: " + str(endstats-end) + "\n")
		file.write("Total Time (per model): " + str(endstats-start) + "\n")
		file.write("Total Time (all models): " + str((endstats-start)*(count+1)) + "\n")
		file.close() 
		
def LFStyleTransferGatys(name, LFdir, loaddir, savedir, num_views, gibbs_loss = False, analysis_only = False, learning_rate = 1e0, epochs = 250, beta = 5, gamma = 500, kappa = 1000, delta=5e7):

	utils.ensure_dir(savedir)
	count = 0
	img_shape = (376,544)
	
	# Load vgg network
	vggdir = loaddir+"vgg16/"
	styledir = loaddir+"styles/"
	vgg = utils.Vgg16()
	vgg.load_state_dict(torch.load(vggdir+"vgg16.weight"))
	if torch.cuda.is_available():
		vgg.cuda()
	
	style_fns = os.listdir(styledir)
	
	for style_fn in style_fns:
	
		currentdir = savedir+"Model"+str(count)+"/"
		utils.ensure_dir(currentdir)
		
		start = time()
		
		if not analysis_only:
		
			# Once for the center
			content_image = utils.tensor_load_rgbimage(LFdir+name+"_0_0_image.png",shape=img_shape)
			content_image = content_image.unsqueeze(0)
			
			if torch.cuda.is_available():
				content_image = content_image.cuda()
			content_image = Variable(utils.preprocess_batch(content_image), requires_grad=True)
				
			optimizer = Adam([content_image], learning_rate)
			mse_loss = torch.nn.MSELoss()
			skip = 50
			
			# Setup loss
			gram_style = cl.get_gram(styledir+style_fn,vgg)
			xc = Variable(content_image.data.clone())
			xc = utils.subtract_imagenet_mean_batch(xc)
			features_xc = vgg(xc)
			f_xc_c = Variable(features_xc[1].data, requires_grad=False)
			
			for k in trange(epochs):
			
				y = content_image
				y = utils.subtract_imagenet_mean_batch(y)

				features_y = vgg(y)
				content_loss = mse_loss(features_y[1], f_xc_c)

				style_loss = 0.
				for m in range(len(features_y)):
					gram_s = Variable(gram_style[m].data, requires_grad=False)
					gram_y = utils.gram_matrix(features_y[m])
					style_loss += mse_loss(gram_y, gram_s[:1, :, :])

				total_loss = content_loss + beta*style_loss
				
				optimizer.zero_grad()
				total_loss.backward(retain_graph=True)
				optimizer.step()

				#if k % skip == 0:
				#	print(k, content_loss.data, style_loss.data, total_loss.data)

			utils.tensor_save_rgbimage(content_image.data[0], currentdir+name+"_0_0.png", torch.cuda.is_available())
			
			center = content_image.clone()
			
			# Once for every other image
			for i in range(-(num_views//2),(num_views//2)+1):
				for j in trange(-(num_views//2),(num_views//2)+1):
						
					if i == 0 and j == 0:
						continue
					
					content_image = utils.tensor_load_rgbimage(LFdir+name+"_"+str(i)+"_"+str(j)+"_image.png",shape=img_shape)
					content_image = content_image.unsqueeze(0)

					if torch.cuda.is_available():
						content_image = content_image.cuda()
					content_image = Variable(utils.preprocess_batch(content_image),requires_grad=True)
					
					# Warp the center based on the disparity maps
					dx = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_dx.npy")
					dy = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_dy.npy")
					mask = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_mask.npy")
					
					dx = utils.resizeNP(dx,img_shape).astype(np.float32)
					dy = utils.resizeNP(dy,img_shape).astype(np.float32)
					mask = utils.resizeNP(mask,img_shape).astype(np.float32)
					
					if gibbs_loss:
						grad_mask = edgeDetect(mask)
						grad_mask = Variable(torch.from_numpy(grad_mask),requires_grad=False)
						if torch.cuda.is_available():
							grad_mask = grad_mask.cuda()
					
					warped_center = utils.warpView(center, dx, dy)
					if torch.cuda.is_available():
						warped_center = warped_center.cuda()
						
					mask = Variable(torch.from_numpy(mask),requires_grad=False)
					if torch.cuda.is_available():
						mask = mask.cuda()

					optimizer = Adam([content_image], learning_rate)
					mse_loss = torch.nn.MSELoss()
					skip = 50
					
					# Setup loss
					xc = Variable(content_image.data.clone())
					xc = utils.subtract_imagenet_mean_batch(xc)
					features_xc = vgg(xc)
					f_xc_c = Variable(features_xc[1].data, requires_grad=False)
					
					for k in range(epochs):
					
						y = content_image
						y = utils.subtract_imagenet_mean_batch(y)

						features_y = vgg(y)
						content_loss = mse_loss(features_y[1], f_xc_c)

						style_loss = 0.
						for m in range(len(features_y)):
							gram_s = Variable(gram_style[m].data, requires_grad=False)
							gram_y = utils.gram_matrix(features_y[m])
							style_loss += mse_loss(gram_y, gram_s[:1, :, :])

						# Remove useless pixels and disparity calculate loss
						pixel_loss = mask*((content_image[0,0]-warped_center[0,0])**2 + (content_image[0,1]-warped_center[0,1])**2 + (content_image[0,2]-warped_center[0,2])**2)
						disparity_loss = gamma * torch.sum(pixel_loss)
					
						total_loss = content_loss + beta*style_loss + gamma*disparity_loss
						
						if gibbs_loss:
							energy = grad_mask*gibbsEnergy(output[0])
							total_loss += kappa*torch.sum(energy)
						
						optimizer.zero_grad()
						total_loss.backward(retain_graph=True)
						optimizer.step()

						#if k % skip == 0:
							#print(k, content_loss.data[0], style_loss.data[0], total_loss.data[0])

					utils.tensor_save_rgbimage(content_image.data[0], currentdir+name+"_"+str(i)+"_"+str(j)+".png", torch.cuda.is_available())
			
		end = time()
			
		lf = loadViewsNumbered(currentdir, num_views, num_views)
		utils.ensure_dir(currentdir[:-1]+"Gantry/")
		saveGantry(lf,currentdir[:-1]+"Gantry/")
		
		stack, foci = getFocalStack(lf, near=0.6, far=1.6, near_step = 0.1, far_step=.15)
		utils.ensure_dir(currentdir[:-1]+"Refocused/")
		saveFocalStack(stack,foci,currentdir[:-1]+"Refocused/")
		
		utils.ensure_dir(currentdir[:-1]+"Visualized/")
		saveScrollMP4(lf, currentdir[:-1]+"Visualized/"+"Model"+str(count)+".mp4")

		lf_original = loadViewsNumbered(LFdir, num_views, num_views, desc="_image")
		style_fn = styledir+os.listdir(styledir)[count]
		loss_matrix = cl.compute_LF_loss(lf_original,lf,vgg,style_fn)
		disp_matrix = cl.compute_LF_disp_loss(lf,name,LFdir)
		cl.save_loss_matrix(loss_matrix,disp_matrix,currentdir[:-1]+"Loss/")
		
		endstats = time()
		
		count += 1	
		
	if not analysis_only:
		file = open(savedir+"Time.txt","w") 
		file.write("Model Time: " + str(end-start) + "\n")
		file.write("Stats Time: " + str(endstats-end) + "\n")
		file.write("Total Time (per model): " + str(endstats-start) + "\n")
		file.write("Total Time (all models): " + str((endstats-start)*(count+1)) + "\n")
		file.close() 

	