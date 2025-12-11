import sys
sys.path.insert(1, r'C:\Users\thijs\University\Year4\Thesis\Pract\open_lth')

import torch
import torch.nn.functional as F
from torchvision import transforms

from open_lth.models import registry

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
import os

import pytorch_grad_cam as gc
from pytorch_grad_cam.utils.image import show_cam_on_image

def calcIoU(heatmap, mask, alpha=3):
    inter = 0.0
    union = 0.0
    for ix, iy, iz in np.ndindex(mask.shape):
        inter += min(heatmap[ix, iy][0], mask[ix, iy][0])
        union += max(heatmap[ix, iy][0], mask[ix,iy][0])
    IoU = inter/union
    L_IoU = 1-IoU
    L_aIoU = (1-(IoU**alpha))/alpha
    return L_IoU, L_aIoU

def getMask(imgFileName, maskFolder):
    mask = Image.open(os.path.join(maskFolder,imgFileName))
    mask = np.asarray(mask).astype("float32")/255
    print(f"Mask: {type(mask)},{mask.shape}")
    return mask

### Produce heatmap ###
def produceHeatmap(model, tensor, targetLayers):
    # TODO: for final version: take constructor out of this function
    explainer = gc.GradCAMPlusPlus(model=model, target_layers=targetLayers, use_cuda=False)
    heatmap = explainer(input_tensor=tensor, targets=None)[0]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
    # print(f"Heatmap: {heatmap.shape} {heatmap.dtype}")
    return heatmap

### Produce heatmap ###
def produceHeatmap2(explainer, tensor):
    # TODO: for final version: take constructor out of this function
    heatmap = explainer(input_tensor=tensor, targets=None)[0]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
    # print(f"Heatmap: {heatmap.shape} {heatmap.dtype}")
    return heatmap

# Output: (PIL, NP) imgs
def tensToImg1(tensor):
    # WORKS
    transTensorToImage = transforms.ToPILImage()
    PILimg = transTensorToImage(tensor)
    NPimg = (np.asarray(PILimg)).astype("float32")/255
    # De-normalize
    # print(f"NPimg: {NPimg.shape} {NPimg.dtype}")
    return PILimg, NPimg

# Output: (NP) img
def tensToImg2(tensor): # c=1 (0) x h=28 (1) x w=28 (2) -> 28 (1) x 28 (2) x 1 (0)
    img = (tensor[0,0]).unsqueeze(-1).repeat(1,1,3).numpy().astype("float32")
    return img

data_transform = transforms.Compose([
    transforms.ToTensor()
    # ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    # This heavily distorts colors 
])

def getNameLabel(tup):
    split = tup[0].split("\\")
    imgName = split[-1]
    labelName = split[-2]
    return imgName, labelName

def loadModel(modelName):
    hparams = registry.get_default_hparams(model_name = modelName)
    hparams.model_hparams.batchnorm_frozen = True
    model = registry.get(hparams.model_hparams, 10)
    # # modelFile = r'C:\Users\thijs\open_lth_data\train_d31ae8c5f55940678c2519351dc81c0c\replicate_1\main\model_ep10_it0.pth'
    # # modelFile = r'C:\Users\thijs\open_lth_data\train_aaaca05da558fe2cb327a82386b7ce8a\replicate_1\main\model_ep10_it0.pth' #85.71%, fc shape: [10,4096] 15ep
    modelFile = r'C:\Users\thijs\open_lth_data\train_db9231e7ef6a7bc1112ae985e7e0d2f8\replicate_1\main\model_ep40_it0.pth' #88.42% fc shape: [10, 4096], 25-30-35ep

    ## Load model
    loaded = torch.load(modelFile)
    model.load_state_dict(loaded)
    model.eval()
    return model

if __name__ == "__main__":
    trainFolder = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\reorga_cropped_dataset\train'
    valFolder = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\reorga_cropped_dataset\val'
    maskFolder = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\reorga_cropped_masks'
    trainData = datasets.ImageFolder(root=trainFolder,transform=data_transform)
    valData = datasets.ImageFolder(root=valFolder,transform=data_transform)


    # trainData.imgs # = list of image paths & labels
    # trainDL.dataset.imgs # = same but through DL
    dataTuples = valData.imgs
    imgPath, inLabIdx = dataTuples[0]
    print(f"Data 0: {dataTuples[0]}")

    model = loadModel(modelName='cifar_resnet_20')
    lossResults = []
    alphaLossResults = []

    stop = 100
    i = 0

    explainer = gc.GradCAMPlusPlus(model=model, target_layers=[model.blocks[8].conv2, model.blocks[8].conv1], use_cuda=False)

    for dataTuple in valData.imgs:
        print(f"Iter {i}")
        if stop:
            if i == stop:
                break
            i += 1
        imageName, labelName = getNameLabel(dataTuple)
        groundImage = Image.open(os.path.join(valFolder, labelName, imageName))
        groundTensor = data_transform(groundImage).unsqueeze(0)
        heatmap = produceHeatmap2(explainer=explainer, tensor=groundTensor)
        mask = getMask(imgFileName=imageName, maskFolder=maskFolder)
        L_IoU, L_aIoU = calcIoU(heatmap=heatmap, mask=mask, alpha=3)
        lossResults.append(L_IoU)
        alphaLossResults.append(L_aIoU)

print(np.average(lossResults))
print(np.average(alphaLossResults))