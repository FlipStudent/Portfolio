import json
from pycocotools.coco import COCO
import data_analyzer as DA

def merge(annFile1, annFile2, saveLocation):
    with open(annFile1, 'r') as f:
        ann_data1 = json.load(f)
    
    with open(annFile2, 'r') as f:
        ann_data2 = json.load(f)   

    new_ann_data = {
        'info': ann_data1['info'],
        'licenses': ann_data1['licenses'],
        'images': ann_data1['images'] + ann_data2['images'],
        'annotations': ann_data1['annotations'] + ann_data2['annotations'],
        'categories': ann_data1['categories']
    }

    with open(saveLocation, 'w') as f:
        json.dump(new_ann_data, f)
        print(f"Selected annotations saved at '{saveLocation}'\n") 
        return saveLocation

if __name__ == "__main__":
    ann1File = r"C:\Users\thijs\University\Year4\Thesis\Pract\Data\annotations\instances_train2017.json" # train
    ann2File = r"C:\Users\thijs\University\Year4\Thesis\Pract\Data\annotations\instances_val2017.json" # val
    saveLocation = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\final_annotations\allAnnotations.json'

    newAnnFile = merge(ann1File, ann2File, saveLocation)
    coco = COCO(newAnnFile)
    DA.analyze(coco)