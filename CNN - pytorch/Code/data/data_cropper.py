import PIL
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import ImageOps
from pycocotools.coco import COCO

def cropImgsToFolder(srcFolder, saveToFolder, newSize):
    images = []
    for filename in os.listdir(srcFolder):
        img = Image.open(os.path.join(srcFolder,filename))
        if img is not None:
            cropImg = ImageOps.pad(img, newSize)
            cropImg.save(os.path.join(saveToFolder,filename))
    return

def createMasksToFolder(coco, saveToFolder, newSize, imgSrcFolder):
    imgIds = coco.getImgIds()
    # imgIds = coco.loadImgs(coco.getImgIds())
    np.random.shuffle(imgIds)
    i = 0
    tL = len(imgIds)
    for imgId in imgIds:
        i+=1
        perc = i/tL
        print(f"{perc}%")
        cocoImg = coco.loadImgs([imgId])[0]
        # print(cocoImg)
        imgName = cocoImg['file_name']
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[imgId]))
        # print(annotations[0])
        # image = Image.open(os.path.join(imgSrcFolder, imgName))
        # plt.imshow(mask)
        # plt.imshow(image)
        # coco.showAnns(annotations)
        # plt.title(annotations[0]['category_id'])
        # plt.show()
        
        # paddedImage = ImageOps.pad(image, newSize)
        # plotImg(paddedImage, "Padded Img")

        mask = createSquarePilMask(coco, annotations[0], newSize, opac=100)
        # mask.convert('RGB')
        mask.save(os.path.join(saveMaskFolder,imgName))
        # plotImg(mask, "Mask function")

        # paddedImage.paste(paddedRgbPilMask, (0,0), paddedRgbPilMask)
        # paddedImage.paste(mask, (0,0), mask)
        # plotImg(paddedImage, f"Output")
    return

def createSquarePilMask(coco, annotation, newSize=(640,640), opac=100):
    npMask = coco.annToMask(annotation)
    rgbPilMask = Image.fromarray(cv2.cvtColor(npMask*255,cv2.COLOR_GRAY2RGB))
    paddedRgbPilMask = ImageOps.pad(rgbPilMask, newSize)
    # paddedRgbPilMask.putalpha(opac)
    return paddedRgbPilMask

def plotImg(img, name):
    plt.imshow(img)
    plt.title(name)
    plt.show()
    return

def showMaskOnImage(mask, image):
    print(image)
    masked = image + 0.2*mask
    plt.imshow(masked)
    plt.show()
    return

if __name__ == "__main__":
    imageSrcFolder = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\final_dataset'
    saveImgFolder = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\cropped_dataset'
    saveMaskFolder = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\reorga_cropped_masks'
    annotationFile = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\final_annotations\ALL_Uni_Fil.json'

    coco = COCO(annotationFile)
    img = coco.getImgIds()
    img = coco.getAnnIds()

    newSize = (640, 640)
    # cropImgsToFolder(imageSrcFolder, saveImgFolder, newSize)
    createMasksToFolder(coco, saveMaskFolder, newSize, imageSrcFolder)