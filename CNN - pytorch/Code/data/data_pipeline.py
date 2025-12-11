from data_analyzer import *
from data_filter import *
from data_univocalizer import *

anno_address1 = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data/annotations/instances_val2017.json'
img_path1 = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data/images/val2017'

anno_address2 = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data/annotations/instances_train2017.json'
img_path2 = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data/images/train2017'

# anno_address2 = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data/annotations/instances_train2017.json'
# img_path2 = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data/images/train2017'

anno_addresses = [anno_address1, anno_address2]
img_paths = [img_path1, img_path2]
new_data_path = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\final_dataset'
new_anno_path = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data\final_annotations'

classes = [3, 10, 44, 47,51,62,67,84]

for _, anno_address in enumerate(anno_addresses):
    start_coco = COCO(anno_address)

    # Filter out unwanted classes
    new_anno_address = filter(start_coco, classes, anno_address, new_anno_path) # (= new_anno_path)

    coco = COCO(new_anno_address)
    analyze(coco)

    # From the remaining classes, find all univocal images & their annotations
    uni_imgs, uni_annos = get_uni(coco)

    # Save those images
    saved_annotations_path = save_annotations(coco, uni_imgs, uni_annos, new_anno_path, new_data_path)
    saved_images_path = save_images(coco, uni_imgs, img_paths[_], new_data_path)

    print(f"FINISHED PROCESSING ({_+1})")

    analyze(COCO(saved_annotations_path))

# saved_annotations_path = r'C:\Users\thijs\University\Year4\Thesis\Pract\Data/filtered_dataset/uni_annotations.json'


# coco = COCO(anno_address)
# print()
# print("---Starting data analysis---")
# analyze(coco)
# print()

# # Now the excess annotations are filtered out
# new_anno_address = filter(classes, anno_address)

# coco = COCO(new_anno_address)

# # Analyze the selected data
# print()
# print("---Selected classes analysis---")
# analyze(coco)
# print()


# # Now univocalize the selected data
# uni = get_uni(coco)

# saved_images_path = save_images(coco, uni, img_path, new_data_path)
# saved_annotations_path = save_annotations(coco, uni, new_anno_address, new_data_path)

# # Analyze the univocalized data
# coco = COCO(saved_annotations_path)
# print()
# print("---Univocal data analysis---")
# analyze(coco)
# print()

# plot_images(coco, uni, bbox=True, seg=False)
