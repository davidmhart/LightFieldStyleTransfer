from scipy.misc import imsave, imresize
import numpy as np
from tqdm import tqdm
from LoadLightField import *
from LightFieldFunctions import *
from SaveLightField import *
from DepthFunctions import *
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm, trange
from time import time

from LightFieldStyleTransferMethods import *
from LightFieldStyleTransferPreprocessor import preprocess

import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
		

	
############################### User Parameters ##################################	
loaddir = "lightfields/"
processdir = "preprocessed_data/"
savedir = "results/"
name = "Swans"
epsilon = 1.4
num_views = 9
modeldir = "StyleNet/"
preprocessing_view = 1 # Put 0 to skip preprocessing, 1 to calibrate based on adjacent view
tuned_k = False
vectorized = True
crop = 5
analysis_only = False # For generated result, regenerate visuals and loss values

# Select method parameters
method = "BP"
fuse_features = True
fuse_images = False
perceptual_loss = False
gibbs_loss = False

##################################################################################

LFdir = processdir + name + "LF/"
ensure_dir(LFdir)

if preprocessing_view:
	preprocess(loaddir, LFdir, name, modeldir, num_views, crop, epsilon, preprocessing_view, tuned_k, vectorized)


# Define save location
folder_name = ""
if fuse_features:
	folder_name += "FuseFeats"
if fuse_images:
	folder_name += "FuseImages"

folder_name += method

if perceptual_loss:
	folder_name += "PerceptualLoss"
if gibbs_loss:
	folder_name += "GibbsLoss"
	
# Complete selected stylization method
start = time()
if method == "BP":
	LFStyleTransferBP(name, LFdir, modeldir, savedir+name+"/"+folder_name+"/", num_views, fuse_features, fuse_images, perceptual_loss, gibbs_loss, analysis_only=analysis_only)
if method == "PostOpt":
	LFStyleTransferPostOpt(name, LFdir, modeldir, savedir+name+"/"+folder_name+"/", num_views, fuse_features, fuse_images, perceptual_loss, gibbs_loss, analysis_only=analysis_only)
if method == "NoOpt":					
	LFStyleTransferNoOpt(name, LFdir, modeldir, savedir+name+"/"+folder_name+"/", num_views, fuse_features, fuse_images, analysis_only=analysis_only)	
if method == "Gatys":
	LFStyleTransferGatys(name, LFdir, modeldir, savedir+name+"/"+folder_name+"/", num_views, gibbs_loss = gibbs_loss, analysis_only = analysis_only)

end = time()
print("Done with style transfer " + folder_name + ". Total time: ", end-start)


