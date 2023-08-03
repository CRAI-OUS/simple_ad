# Simple AD
The pytorch implementation of the SFCN model along with the weights for the paper: A minimalistic approach to classifying Alzheimer’s disease using simple and extremely small convolutional neural networks

## Dataset
Since ADNI is a restricted dataset, we cannot share the dataset. However, we have provided the similar code to extract data from the IXI dataset to demonstrate the process. 

## Usage
Start by running the preprocessing script to download and preprocess IXI
Then run the inference script to run prediction on the preprocessed data. As IXI consist of healthy subjects only, the prediction will be low for most subjects.