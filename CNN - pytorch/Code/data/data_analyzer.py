from pycocotools.coco import COCO
import numpy as np

def analyze(coco):
    other_classes = np.zeros(90)
    self_class = np.zeros(90)

    largestW = 0
    saveLW = [0,0]
    largestH = 0
    saveLH = [0,0]
    smallestW = 9999
    saveSW = [0,0]
    smallestH = 9999
    saveSH = [0,0]
    totalPixels = 0
    nImages = 0
    # Through all images
    for imgID in coco.getImgIds():
        img = coco.loadImgs(ids=[imgID])[0]
        h = int(img['height'])
        w = int(img['width'])
        if h > largestH:
            largestH = h
            saveLH = [h, w]
        if w > largestW:
            largestW = w
            saveLW = [h, w]
        if h < smallestH:
            smallestH = h
            saveSH = [h, w]
        if w < smallestW:
            smallestW = w
            saveSW = [h, w]
        totalPixels += (h*w)
        nImages += 1
        annoIDs = coco.getAnnIds(imgIds=imgID)
        x = np.array(annoIDs)
        other_annos = len(np.unique(x))-1
        for annoID in annoIDs:
            annotation = coco.loadAnns(annoID)[0]
            catID = annotation['category_id']
            self_class[catID-1] += 1
            other_classes[catID-1] += other_annos

    for catID in coco.getCatIds():
        cat = coco.loadCats(catID)[0]
        s = self_class[catID-1]
        o = other_classes[catID-1]
        print(f"{cat['id']}, {cat['name']}, S:, {s}, O:, {o}, {s/o}")

    print(f"Total annotations: {len(coco.getAnnIds())}")
    print(f"Total images: {len(coco.getImgIds())}")

    print(f"Img with largest height: shape={saveLH} {saveLH[0]*saveLH[1]}")
    print(f"Img with largest width: shape={saveLW} {saveLW[0]*saveLW[1]}")
    print(f"Img with smallest height: shape={saveSH}")
    print(f"Img with smallest width: shape={saveSW}")
    print(f"Average pixels: {totalPixels/nImages}")

if __name__ == "__main__":
    # annFile = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\annotations\trial_val.json'
    annFile = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\final_annotations\ALL_Uni_Fil.json'
    coco = COCO(annFile)
    analyze(coco)