import os

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

from scipy.interpolate import RectBivariateSpline

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def toGrayScale(image):
    red = image[0,:,:]
    green = image[1,:,:]
    blue = image[2,:,:]
    result = 0.299*red + 0.557*green + 0.114*blue
    return result
	
def generalConvolution(image,kernel):

	conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
	conv.weight = nn.Parameter(torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0))
	
	if torch.cuda.is_available():
		conv.cuda()
	
	result=conv(image)

	return result
	
def gaussianBlur(image):

	image = image.clone()
	image = image.unsqueeze(0).unsqueeze(0)

	kernel = np.array([[1, 2, 1],[2,4,2],[1,2,1]])
	result = 1/16 * generalConvolution(image, kernel)
	
	return torch.squeeze(result)
		
	
def edgeDetect(image):
    
	image = image.clone()
	image = toGrayScale(image.squeeze())
	image = image.unsqueeze(0).unsqueeze(0)

	kernel_x = np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
	result_x = 1/8 * generalConvolution(image, kernel_x)

	kernel_y = np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
	result_y = 1/8 *generalConvolution(image, kernel_y)
	#G_y=conv2(Variable(x)).data.view(1,256,512)

	result=torch.sqrt(torch.pow(result_x,2)+ torch.pow(result_y,2))
	
	return torch.squeeze(result)
		
def warpView(content,dx,dy):
	
	#print(content)
	
	content = content.data[0].cpu().numpy()
	
	#print(content)
	
	_, rows, cols = content.shape
	
	# Generate interpolated field
	x, y = np.array(range(cols)),np.array(range(rows))
	xv, yv = np.meshgrid(x,y)
	
	# Warp left image to the right side view
	ipR = RectBivariateSpline(y, x, content[0,:,:], kx=1, ky=1)
	ipG = RectBivariateSpline(y, x, content[1,:,:], kx=1, ky=1)
	ipB = RectBivariateSpline(y, x, content[2,:,:], kx=1, ky=1)

	warped = np.zeros((3,rows,cols))
	warped[0,:,:] = ipR(yv+dy,xv+dx,grid=False)
	warped[1,:,:] = ipG(yv+dy,xv+dx,grid=False)
	warped[2,:,:] = ipB(yv+dy,xv+dx,grid=False)
	#warped[0,:,:] = np.multiply(ipR(yv+dy,xv+dx,grid=False),mask)
	#warped[1,:,:] = np.multiply(ipG(yv+dy,xv+dx,grid=False),mask)
	#warped[2,:,:] = np.multiply(ipB(yv+dy,xv+dx,grid=False),mask)
	
	warped = np.expand_dims(warped,axis=0)
	
	return Variable(torch.from_numpy(warped.astype(np.float32)), requires_grad=False)

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
	
def warpFeatures(content,dx,dy):
	
	content = content.data[0].cpu().numpy()
	
	ch, rows, cols = content.shape
	warped = np.zeros((ch,rows,cols))
	
	for i in range(ch):
	
		# Generate interpolated field
		x, y = np.array(range(cols)),np.array(range(rows))
		xv, yv = np.meshgrid(x,y)
		
		# Warp left image to the right side view
		ipC = RectBivariateSpline(y, x, content[i,:,:], kx=1, ky=1)
		
		warped[i,:,:] = ipC(yv+dy,xv+dx,grid=False)
		
	warped = np.expand_dims(warped,axis=0)
	
	return Variable(torch.from_numpy(warped.astype(np.float32)), requires_grad=False)

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
	
class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h

        return [relu1_2, relu2_2, relu3_3, relu4_3]


def tensor_from_numpy(img, shape=None, scale=None):
    if shape is not None:
        img = img.resize((shape[1], shape[0]), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
	
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img
		
		
def tensor_load_rgbimage(filename, shape=None, scale=None):
    img = Image.open(filename)
    if shape is not None:
        img = img.resize((shape[1], shape[0]), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
	
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def subtract_imagenet_mean_batch(batch):
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    if torch.cuda.is_available():
        mean = mean.cuda()
    batch = batch.sub(Variable(mean))
    return batch


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch


