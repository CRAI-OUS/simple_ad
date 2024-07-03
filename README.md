# Simple AD
Simple Fully Convolutional Network (SFCN) trained on Alzheimer's disease classification using the ADNI dataset. 

**A minimalistic approach to classifying Alzheimer’s disease using simple and extremely small convolutional neural networks**
*Edvard O. S. Grødem, Esten Leonardsen, Bradley J. MacIntosh, Atle Bjørnerud, Till Schellhorn, Øystein Sørensen, Inge Amlien, Anders M. Fjell, for the Alzheimer’s Disease Neuroimaging Initiative*

# Models
Three models are provided in this repository:
1. SFCN (our implementation) [1]
2. EfficientNet ([Monai implementation](https://docs.monai.io/en/stable/networks.html#efficientnetbn)) [2]
3. DenseNet ([Monai implementation](https://docs.monai.io/en/stable/networks.html#densenet)) [3]
## Dataset
Since ADNI is a restricted dataset, we cannot share the dataset. However, we have provided the similar code to extract data from the IXI dataset to demonstrate the process. 

## Usage
Start by running the preprocessing script to download and preprocess IXI.
Then run the inference script to run prediction on the preprocessed data. As IXI consist of only healthy subjects, the prediction will be low for most subjects.'

## Model Hyperparamteres
The models weights are trained with the following hyperparameters:
| Model Type   | Learning Rate | Weight Decay | Optimizer | Batch Size |
|--------------|---------------|--------------|-----------|------------|
| EfficientNet | 0.001         | 0.01         | AdamW     | 4          |
| DenseNet     | 0.005         | 0.01         | SGD       | 4          |
| SFCN         | 0.005         | 0.1          | SGD       | 4          |

## References
[1] [Accurate brain age prediction with lightweight deep neural networks.](https://doi.org/10.1016/j.media.2020.101871) Peng, H., Gong, W., Beckmann, C. F., Vedaldi, A., & Smith, S. M. (2021). In *Medical Image Analysis*, 68, 101871. Elsevier.

[2] [EfficientNet: Rethinking model scaling for convolutional neural networks.](https://proceedings.mlr.press/v97/tan19a.html) Tan, M., & Le, Q. (2019). In *International Conference on Machine Learning* (pp. 6105-6114). PMLR.

[3] [Densely connected convolutional networks.](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html)Huang, G., Liu, Z., Van Der Maaten, L., Weinberger, K.Q., (2017). In *Proceedings of the IEEE conference
on computer vision and pattern recognition* (pp. 4700–4708)