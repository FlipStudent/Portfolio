from pycocotools.coco import COCO
import json
import data_analyzer

def filter(coco, keepCatArray, annFile, newAnnLoc=r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\final_annotations'):
    saveLocation = newAnnLoc + r'\ALL_noUni_Fil.json'
    print()
    print(f"Filtering out unwanted classes, keeping: {keepCatArray}")
    annoIDs = coco.getAnnIds(catIds=keepCatArray)
    annotations = coco.loadAnns(annoIDs)

    keep_anno_ID = [anno['id'] for anno in annotations]
    keep_anno = coco.loadAnns(keep_anno_ID)

    with open(annFile, 'r') as f:
        ann_data = json.load(f)      
    
    keep_img_IDs = []
    for catID in keepCatArray:
        res = coco.getImgIds(catIds = [catID])
        keep_img_IDs += res
    print(f"Keeping {len(keep_img_IDs)}")

    new_ann_data = {
        'info': ann_data['info'],
        'licenses': ann_data['licenses'],
        'images': coco.loadImgs(keep_img_IDs),
        'annotations': [anno for anno in keep_anno if anno['category_id'] in keepCatArray],
        'categories': [cat for cat in ann_data['categories'] if cat['id'] in keepCatArray] 
    }

    print(f"Number of annotations in old annotation file: {len(coco.getAnnIds())}")
    print(f"Number of images in old annotation file: {len(coco.getImgIds())}")
    print(f"Number of images in filtered annotation file: {len(new_ann_data['images'])}")
    print(f"Number of annotations in filtered annotation file: {len(new_ann_data['annotations'])}")

    # Save the modified annotation file to a new JSON file
    with open(saveLocation, 'w') as f:
        json.dump(new_ann_data, f)
        print(f"Selected annotations saved at '{saveLocation}'") 
        print()
    return saveLocation

if __name__ == "__main__":
    # dataDir = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data'
    new_ann_dir = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\final_annotations'
    annFile = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data/final_annotations/uni_allAnnotations.json'
    coco = COCO(annFile)
    new_anno_address = filter(coco, [3,10,44,47,51,62,67,84], annFile, new_ann_dir)
    newCoco = COCO(new_anno_address)
    data_analyzer.analyze(newCoco)