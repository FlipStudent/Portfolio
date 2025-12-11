# RUG Deep Learning - Group 12

## Setup
We recommend using a virtual environment for installing dependencies. Dependencies can be install using:
```
pip install -r requirements.txt
```
The [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) dataset can be found on Kaggle.
Extracted folder should be named Fruits-360/.

## Training and testing
Training the model on the fruits dataset can be done by using:
```
python train.py
```
Testing all 5 trained models on the fruits test set can be done using:
```
python test.py
```
If any visitors simply want to test functionality, there is a dummy dataset to use. The dummy dataset is a smaller set of images from Fruits-360.
Fruits-360 still needs to be downloaded and extracted. The dummy dataset can be used by uncommenting relevant
commands at the bottom of `train.py` and `test.py`.
```
train(file="utils/dummy_fruits.csv")
test(file="utils/dummy_fruits.csv")
generate_reconstructions(file="utils/dummy_fruits.csv")
```

## Output
The `output/` folder on GitHub contains the trained models in `models/` as well as training, validation and testing logs in `logs/`. In addition,
the main `output/` folder contains images of acquired results.

## Workflow
Create branch for a specific issue. Push to that branch freely. Merge with main through pull request.
