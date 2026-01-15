import numpy as np
import tifffile
from cellpose import models





# Load image
img = tifffile.imread("/data3/love/lung_cancer_BOMI2/image_data/raw_image_data/all/BOMI2_TIL_1_Core[1,1,A]_[5091,35249]_component_data.tif")

# Extract DAPI and CK channels
dapi = img[0,:,:]  # assuming DAPI is channel 0
ck = img[6,:,:]    # assuming CK is channel 1


model = models.Cellpose(model_type='nuclei')
masks, flows, styles, diams = model.eval(dapi, diameter=None, channels=[0,0])
