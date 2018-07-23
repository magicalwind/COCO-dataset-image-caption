# COCO dataset image caption

## About the COCO dataset

1. ### Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. ### Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. ### Download some specific data from here: http://cocodataset.org/#download (described below)

## About the files:

- Dataset.ipynb

Visualize the data, including the image, caption pairs. 

- data_loader.py, vocabulary.py
Build the needed class for data loading, and the vocabulary class. Based on the code of [yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning). Modify the code for batching processing.

- model.py

Main structure of the Neural Network. Build a CNN encoder with pre-trained Resnet50 for training, freeze the pre_trained paramenters and replace the final layer with an embedding layer. Concat the extracted image vector with caption vectors and feed them into decoder. In RNN decoder, implement forward method and Sample method with LSTM. Use greedy search to create a sentence word by word based on the hidden state and the output word vector from the last time step.  

- Preview.ipynb

Integrate the data_load.py, vocabulary.py. Tokenize the sentences with nltk. Build an iterator to load data in batches where the sentence length in each batch is equal. Test the shape of the output of encoder and decoder.

- Training.ipynb

Set the training parameters and start a training. (It takes time on training at least on my desktop) Save the trained model after each epoch.

- Generating Captions

Predicting the captions with the sample method created in Decoder. Convert the index sequences to word sequences and view the output. 
