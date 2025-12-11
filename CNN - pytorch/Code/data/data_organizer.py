from pycocotools.coco import COCO
import numpy as np
import os
from data_univocalizer import save_images

inputAnnotationFile = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\final_annotations\ALL_Uni_Fil.json'
inputImageFolder = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\cropped_dataset'
outputTrainFolder = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\reorga_cropped_dataset\train'
outputValFolder = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\reorga_cropped_dataset\val'
# outputTestFolder = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\reorganized_dataset'
np.random.seed(333)

def prepareFolders(coco):
    cats = coco.loadCats(coco.getCatIds())
    for cat in cats:
        trainFolder = outputTrainFolder + '/' + cat['name']
        valFolder = outputValFolder + '/' + cat['name']
        if not os.path.isdir(trainFolder):
            os.mkdir(trainFolder)
        if not os.path.isdir(valFolder):
            os.mkdir(valFolder)

coco = COCO(inputAnnotationFile)
prepareFolders(coco)

def splitClass(coco, catId, trainRatio=0.8, valRatio=0.2):
    if trainRatio + valRatio != 1:
        print("Split ratios do not add up to 1")
        return
    classImgIds = coco.getImgIds(catIds=[catId])

    np.random.shuffle(classImgIds)

    imgsInClass = len(classImgIds)
    cutOffIdx = int(imgsInClass * trainRatio)
    train, val = classImgIds[:cutOffIdx], classImgIds[cutOffIdx:]
    return train, val


allCatIds = coco.getCatIds()
for catId in allCatIds:
    trainIds, valIds = splitClass(coco, catId)
    cat = coco.loadCats(catId)[0]
    trainFolder = outputTrainFolder + '/' + cat['name']
    valFolder = outputValFolder + '/' + cat['name']
    save_images(coco, coco.loadImgs(trainIds), inputImageFolder, trainFolder)
    save_images(coco, coco.loadImgs(valIds), inputImageFolder, valFolder)
    # allCatIds = coco.getCatIds()
    # for catId in allCatIds:


# allImgIds = coco.getImgIds()
# allImgs = coco.loadImgs(allImgIds)

