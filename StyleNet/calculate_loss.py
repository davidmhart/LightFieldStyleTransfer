from StyleNet import utils

import torch
from torch.autograd import Variable

import numpy as np
from skimage.transform import resize

import matplotlib.pyplot as plt
	
def getView(lf, ic, jc):

	h, w, _, _, ch = lf.shape
	
	i = (h-1)//2 + ic
	j = (w-1)//2 + jc
	
	return lf[i,j,:,:,:3]
	
def get_gram(style_fn, vgg, shape=(256,256)):
	style = utils.tensor_load_rgbimage(style_fn, shape)
	style = style.repeat(1, 1, 1, 1)
	style = utils.preprocess_batch(style)
	if torch.cuda.is_available():
		style = style.cuda()
	
	if torch.cuda.is_available():
		style_v = Variable(style).cuda()
	else:
		style_v = Variable(style)
	style_v = utils.subtract_imagenet_mean_batch(style_v)
	features_style = vgg(style_v)
	gram_style = [utils.gram_matrix(y) for y in features_style]
	
	return gram_style

	
def compute_perceptual_loss(im_original,im_style,vgg,gram_style,tensor=False):

	mse_loss = torch.nn.MSELoss()
	
	if torch.cuda.is_available():
		y = Variable(utils.preprocess_batch(utils.tensor_from_numpy(im_original)).unsqueeze(0)).cuda()
		xc = Variable(utils.preprocess_batch(utils.tensor_from_numpy(im_style)).unsqueeze(0)).cuda()
	else:
		y = Variable(utils.preprocess_batch(utils.tensor_from_numpy(im_original)).unsqueeze(0))
		xc = Variable(utils.preprocess_batch(utils.tensor_from_numpy(im_style)).unsqueeze(0))

	#print(y.size())
	#print(xc.size())

	y = utils.subtract_imagenet_mean_batch(y)
	xc = utils.subtract_imagenet_mean_batch(xc)

	features_y = vgg(y)
	features_xc = vgg(xc)

	f_xc_c = Variable(features_xc[1].data, requires_grad=False)

	content_loss = mse_loss(features_y[1], f_xc_c)

	style_loss = 0.
	for m in range(len(features_y)):
		gram_s = Variable(gram_style[m].data, requires_grad=False)
		gram_y = utils.gram_matrix(features_y[m])
		style_loss += mse_loss(gram_y, gram_s[:1, :, :])
		
	if tensor:
		return content_loss, style_loss
	else:
		return content_loss.data.cpu().numpy(),style_loss.data.cpu().numpy()
	

	
def compute_LF_loss(lf_original,lf_style,vgg,style_fn):

	gram_style = get_gram(style_fn,vgg)
	
	im_h,im_w,rows,cols,_ = lf_style.shape
	
	loss_matrix = np.zeros((im_h,im_w,2))
	
	for i in range(im_h):
		for j in range(im_w):
			im_style = lf_style[i,j,:,:,:3]
			im_original = resize(lf_original[i,j,:,:,:3],(rows,cols),mode="constant",anti_aliasing=False)
			
			result = compute_perceptual_loss(im_original,im_style,vgg,gram_style)
			loss_matrix[i,j,0] = result[0]
			loss_matrix[i,j,1] = result[1]
	
	return loss_matrix

	
def compute_disp_loss(lf,i,j,name,LFdir):

	h,w,rows,cols,_ = lf.shape
	
	center = getView(lf,0,0)
	
	output = getView(lf,i,j)

	dx = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_dx.npy")
	dy = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_dy.npy")
	mask = np.load(LFdir + name + "_" + str(i) + "_" + str(j) + "_mask.npy")
	
	dx = utils.resizeNP(dx,(rows,cols)).astype(np.float32)
	dy = utils.resizeNP(dy,(rows,cols)).astype(np.float32)
	mask = utils.resizeNP(mask,(rows,cols)).astype(np.float32)
	
	warped_center = utils.warpViewNP(center, dx, dy)
	
	#loss = np.sum(np.expand_dims(mask,axis=2)*(np.abs(lf[i,j]-warped_center)))
	
	pixel_loss = mask*((output[:,:,0]-warped_center[:,:,0])**2 + (output[:,:,1]-warped_center[:,:,1])**2 + (output[:,:,2]-warped_center[:,:,2])**2)
	loss = np.sum(pixel_loss)
	
	return loss
		
		
def compute_LF_disp_loss(lf,name,LFdir):

	lf = lf[:,:,:,:,:3]

	h,w,_,_,_ = lf.shape
	
	disp_loss_matrix = np.zeros((h,w))
	
	for i in range(-(h//2),(h//2)+1):
		for j in range(-(w//2),(w//2)+1):
			if i==0 and j==0:
				continue
				
			disp_loss_matrix[i+h//2,j+w//2] = compute_disp_loss(lf,i,j,name,LFdir)
			
	return disp_loss_matrix
	
	
def save_loss_matrix(loss_matrix,disp_matrix,savedir):
	
	utils.ensure_dir(savedir)
	
	plt.imshow(loss_matrix[:,:,0],cmap="jet");plt.colorbar();plt.title("Content_Loss");plt.savefig(savedir+"Content_Loss")
	plt.clf()
	
	plt.imshow(loss_matrix[:,:,1],cmap="jet");plt.colorbar();plt.title("Style_Loss");plt.savefig(savedir+"Style_Loss")
	plt.clf()
	
	plt.imshow(disp_matrix,cmap="jet");plt.colorbar();plt.title("Disparity_Loss");plt.savefig(savedir+"Disparity_Loss")
	plt.close()
	
	#alpha = 1e4
	#plt.imshow(alpha*loss_matrix[:,:,0] + loss_matrix[:,:,1],cmap="jet");plt.colorbar();plt.title("Perceptual_Loss (alpha = " + str(alpha) + ")");plt.savefig(savedir+"Perceptual_Loss")
	#plt.close()
	
	#gamma = 500
	#plt.imshow(alpha*loss_matrix[:,:,0] + loss_matrix[:,:,1] + gamma*disp_matrix,cmap="jet");plt.colorbar();plt.title("Total_Loss (gamma = " + str(gamma) + ")");plt.savefig(savedir+"Total_Loss")
	#plt.close()
	
	np.save(savedir+"loss_matrix", np.concatenate((loss_matrix,np.expand_dims(disp_matrix,axis=2)),axis=2))
	
	file = open(savedir+"Losses.txt","w") 

	file.write("Content Loss: " + str(np.sum(loss_matrix[:,:,0])) + "\n") 
	file.write("Style Loss: " + str(np.sum(loss_matrix[:,:,1])) + "\n") 
	file.write("Disparity Loss: " + str(np.sum(disp_matrix)) + "\n")
	file.write("Perceptual Loss: " + str(np.sum(loss_matrix)) + "\n")
	file.write("Total Loss: " + str(np.sum(loss_matrix)+np.sum(disp_matrix))) 

	file.close() 


			