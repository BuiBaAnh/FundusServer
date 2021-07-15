from PIL import Image
import numpy as np
from Model.Utils.transforms import *
from Model.Models.model_ROI import *
from Model.Models.model_OD import *
from Model.Models.model_OC import *
from Model.Utils.utils import *
from Model.Utils.metrics import *
import torch.nn.functional as F
import torch
from PIL import Image

from skimage.measure import label, regionprops
from skimage import morphology

from io import BytesIO
import base64



def predictODC(img, phase) :

    ROI_SIZE = 640
    CROP_SIZE = 512

    load_from = "./Model/Weights/deeplabv3ODForROI.pth"
    
    if (phase == "OD"):
        load_from_Seg = './Model/Weights/deeplabv3ODResnet101.pth'
        num_filter = 81
        thresh = 0.99
    if (phase == "OC"):
        load_from_Seg = './Model/Weights/deeplabv3OCResnet101.pth'
        num_filter = 21
        thresh = 0.95

    img = np.array(Image.open(img))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = OpticsROI(outputchannels=1)
    model = model.to(device)

    model.load_state_dict(torch.load(load_from, map_location=device))
    model.eval()   # Set model to the evaluation mode

    modelOD = OpticsDisc(outputchannels=1)
    modelOD = modelOD.to(device)

    modelOD.load_state_dict(torch.load(load_from_Seg, map_location=device))
    modelOD.eval()   # Set model to the evaluation mode


    # load image
    org_img = img
    # org_img = cv2.cvtColor(org_img,cv2.COLOR_BGR2RGB)

    # Disc region detection by U-Net
    temp_img = cv2.resize(org_img, (ROI_SIZE, ROI_SIZE))
    temp_img = transforms_train(temp_img)
    temp_img = torch.unsqueeze(temp_img, dim = 0)
    pred = model(temp_img.to(device))['out']
    pred = F.sigmoid(pred)
    pred = pred[0][0].data.cpu().numpy()*255
    pred = pred.astype(np.uint8)
    # pred = Image.fromarray(pred)
    # pred
    disc_map = BW_img(pred, 1)
    regions = regionprops(label(disc_map))
    C_x = int(regions[0].centroid[0] * org_img.shape[0] / ROI_SIZE)
    C_y = int(regions[0].centroid[1] * org_img.shape[1] / ROI_SIZE)

    # ''' get disc region'''
    disc_region, err_coord, crop_coord = disc_crop(org_img, CROP_SIZE, C_x, C_y)
    disc_region_img = disc_region.astype(np.uint8)

    disc_region_img_for_predict = torch.unsqueeze(transforms_train(disc_region_img), dim = 0)
    OD = modelOD(disc_region_img_for_predict.to(device))['out']
    OD = F.sigmoid(OD)
    OD[OD >= thresh] = 1
    OD[OD < thresh] = 0
    OD = OD[0][0].data.cpu()*255

    #Postprocessing
    OD = scipy.signal.medfilt2d(OD,num_filter)
    OD = morphology.binary_erosion(OD, morphology.rectangle(2,1))
    OD = get_largest_fillhole(OD)*255.0

    OD = torch.from_numpy(OD)
    ret_img = torch.zeros(org_img.shape)
    h0,h1, w0, w1 = crop_coord
    ret_img[h0:h1, w0:w1,0] = OD
    ret_img[h0:h1, w0:w1,1] = OD
    ret_img[h0:h1, w0:w1,2] = OD
    ret_img = ret_img.numpy().astype(np.uint8)


    ret = Image.fromarray(ret_img)
    data = BytesIO()
    ret.save(data, "PNG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return encoded_img_data.decode('utf-8')

def predictODCForExp(img, phase) :

    ROI_SIZE = 640
    CROP_SIZE = 512

    load_from = "./Model/Weights/deeplabv3ODForROI.pth"
    
    if (phase == "OD"):
        load_from_Seg = './Model/Weights/deeplabv3ODResnet101.pth'
        num_filter = 81
        thresh = 0.99
    if (phase == "OC"):
        load_from_Seg = './Model/Weights/deeplabv3OCResnet101.pth'
        num_filter = 21
        thresh = 0.95

    img = np.array(Image.open(img))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = OpticsROI(outputchannels=1)
    model = model.to(device)

    model.load_state_dict(torch.load(load_from, map_location=device))
    model.eval()   # Set model to the evaluation mode

    modelOD = OpticsDisc(outputchannels=1)
    modelOD = modelOD.to(device)

    modelOD.load_state_dict(torch.load(load_from_Seg, map_location=device))
    modelOD.eval()   # Set model to the evaluation mode


    # load image
    org_img = img
    # org_img = cv2.cvtColor(org_img,cv2.COLOR_BGR2RGB)

    # Disc region detection by U-Net
    temp_img = cv2.resize(org_img, (ROI_SIZE, ROI_SIZE))
    temp_img = transforms_train(temp_img)
    temp_img = torch.unsqueeze(temp_img, dim = 0)
    pred = model(temp_img.to(device))['out']
    pred = F.sigmoid(pred)
    pred = pred[0][0].data.cpu().numpy()*255
    pred = pred.astype(np.uint8)
    # pred = Image.fromarray(pred)
    # pred
    disc_map = BW_img(pred, 1)
    regions = regionprops(label(disc_map))
    C_x = int(regions[0].centroid[0] * org_img.shape[0] / ROI_SIZE)
    C_y = int(regions[0].centroid[1] * org_img.shape[1] / ROI_SIZE)

    # ''' get disc region'''
    disc_region, err_coord, crop_coord = disc_crop(org_img, CROP_SIZE, C_x, C_y)
    disc_region_img = disc_region.astype(np.uint8)

    disc_region_img_for_predict = torch.unsqueeze(transforms_train(disc_region_img), dim = 0)
    OD = modelOD(disc_region_img_for_predict.to(device))['out']
    OD = F.sigmoid(OD)
    OD[OD >= thresh] = 1
    OD[OD < thresh] = 0
    OD = OD[0][0].data.cpu()*255

    #Postprocessing
    OD = scipy.signal.medfilt2d(OD,num_filter)
    OD = morphology.binary_erosion(OD, morphology.rectangle(2,1))
    OD = get_largest_fillhole(OD)*255.0

    OD = torch.from_numpy(OD)
    ret_img = torch.zeros(org_img.shape)
    h0,h1, w0, w1 = crop_coord
    ret_img[h0:h1, w0:w1,0] = OD
    ret_img[h0:h1, w0:w1,1] = OD
    ret_img[h0:h1, w0:w1,2] = OD
    ret_img = ret_img.numpy().astype(np.uint8)


    ret = Image.fromarray(ret_img)
    return ret