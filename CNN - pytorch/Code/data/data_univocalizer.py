from pycocotools.coco import COCO
import json
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import requests
import numpy as np

# file = r'instances_train2017.json'
# file = r'instances_train2017.json'

def save_images(completeCoco, uni, img_path, new_data_path):
    # Create dir to save filtered data & labels to
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)
    
    # Now save the imgs
    for img in uni:
        # Load the image data using the COCO API
        img_data = completeCoco.loadImgs(img['id'])[0]
        # Load the image file using cv2
        img_file = '{}/{}'.format(img_path, img_data['file_name'])
        image = cv2.imread(img_file)
        # Save the image file to the new directory
        save_path = '{}/{}'.format(new_data_path, img_data['file_name'])
        cv2.imwrite(save_path, image)
    print(f"---Saved images at '{new_data_path}'---")
    return(new_data_path)
    
# saved_annotations_path = save_annotations(coco, uni_imgs, uni_annos, new_anno_path, new_data_path)
# def save_annotations(coco, uni_imgs, uni_annos, anno_path, newAnnoName, new_data_path, prevAnnoFileName):
def save_annotations(completeCoco, uni_imgs, uni_annos, oldAnnoFile, newAnnoFile):
    if os.path.exists(newAnnoFile): newAnnoFile += 'x'

    with open(oldAnnoFile, 'r') as f:
        annoData = json.load(f)

    toSaveImageIDs = [img['id'] for img in uni_imgs]
    toSaveAnnoIDs = [anno['id'] for anno in uni_annos]

    newAnnData = {
        'info': annoData['info'],
        'licenses': annoData['licenses'],
        'images': completeCoco.loadImgs(toSaveImageIDs),
        'annotations': completeCoco.loadAnns(toSaveAnnoIDs),
        'categories': annoData['categories']
    }

    # Write the filtered annotations to a new JSON file
    with open(newAnnoFile, 'w') as f:
        json.dump(newAnnData, f)

    print(f"---Saved annotations at '{newAnnoFile}'---")
    print(f"Total images: {len(newAnnData['images'])}")
    print(f"Total annotations: {len(newAnnData['annotations'])}")
    return(newAnnoFile)
    

def plot_images(coco, uni_array, bbox=False, seg=False):

    for img in uni_array:

        plot_img = img
        plot_img_id = plot_img['id']
        img_info = coco.loadImgs([plot_img_id])[0]
        print(img.keys())

        img_url = img_info['coco_url']
        px_img = Image.open(requests.get(img_url, stream=True).raw)
        plt.title(img['name'])
        plt.imshow(np.asarray(px_img))

        anno_ids = coco.getAnnIds(imgIds=plot_img_id)
        img_anno = coco.loadAnns(anno_ids)
        coco.showAnns(img_anno, draw_bbox=True)
        plt.show()

def get_uni(coco):
    cats = coco.loadCats(coco.getCatIds())
    catNames = []
    for cat in cats:
        catNames.append(cat['name'])

    # get the image IDS
    imgIDs = coco.getImgIds()
    # Load the images
    images = coco.loadImgs(imgIDs)

    # Each annotation ID represents a certain label (with a bunch of information)
    # We want univocal data, so only allow annotation arrays with a length of 1
    uni_imgs = [img for img in images if len(coco.getAnnIds(imgIds=img['id']))==1]
    uni_annos = []
    for img in uni_imgs:
        uni_annos += coco.getAnnIds(imgIds = [img['id']])
    print("")
    print(f"Univocalized images with cat {catNames}")
    print(f"N of univocal images: {len(uni_imgs)}")
    print(f"N of univocal annotations: {len(uni_annos)}")
    print("")
    return(uni_imgs, coco.loadAnns(uni_annos))

if __name__ == "__main__":
    annFile = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\final_annotations\ALL_noUni_Fil.json'
    # annoFolder = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\final_annotations'

    # new_data_path = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\final_dataset'
    newAnnoFile = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\final_annotations\ALL_Uni_Fil.json'

    coco = COCO(annFile)

    # data_path = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data'
    # toSaveFolder = data_path + r'/final_annotations'
    # img_path = data_path + r'/images/val2017'

    uni_imgs, uni_annos = get_uni(coco)
    # plot_images(coco, uni, bbox=True, seg=False)
    # saved_images_path = save_images(coco, uni_imgs, img_path, new_data_path)
    saved_annotations_path = save_annotations(completeCoco = coco, 
                                              uni_imgs = uni_imgs, 
                                              uni_annos = uni_annos, 
                                              oldAnnoFile = annFile,
                                              newAnnoFile = newAnnoFile
                                              )
    # saved_annotations_path = save_annotations(coco = coco, 
    #                                           uni_imgs = uni_imgs, 
    #                                           uni_annos = uni_annos, 
    #                                           anno_path = annoFolder, 
    #                                           newAnnoName = 'uni_allAnnotations', 
    #                                           annoSaveFolder = toSaveFolder
    #                                           )

# N/uni in Val: 593 / 5000
# N/uni in Train: 14,486 / 118,287