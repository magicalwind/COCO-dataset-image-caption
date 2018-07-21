# COCO dataset image caption

# About the COCO dataset

1. ## Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. ## Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. ## Download some specific data from here: http://cocodataset.org/#download (described below)

4. ## About the files:

- Preview.ipynb
Visualize the data, including the image, caption pairs. 
- data_loader.py, vocabulary.py
Build the needed class for data loading, and the vocabulary class. Based on the code of [yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)
-
