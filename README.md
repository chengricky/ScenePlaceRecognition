# The Unified Network for Scene Description and Scene Classification

The code achieves the unified network with two branches for scene classification and scene description (NetVLAD). The two branches of the network are trained separately on the classification dataset (Places-365) and the description dataset (Pittsburgh). The trained networks are combined together to test on multiple real-world datasets.

If you use this code, please refer to the paper:
[Unifying Visual Localization and Scene Recognition for Assistive Navigation](http://doi.org/10.1109/ACCESS.2020.2984718)


The directory structure of this repository is:
```
|--ScenePlaceRecognition
|----DataSet
|----Place365
|----UnifiedModel

```

In this code, the multiple backbones (Wide ResNet-18, MobileNet V2 and ShuffleNet V2, actually more as you modify) are supported in the unified network. 

The usage of the training and testing code.

`AttentionRetrieval.py` trains the NetVLAD network (with data augmentation).

`DimReduction.py` trains the dimension reduction layer (PCA with whitening) based on NetVLAD. 

`Place365/train_PlacesCNN.py` trains the scene classification network.

`ScenePlaceRecognitionMain.py` tests the unified network on different datasets, meanwhile saves the results of scene classification and scene descriptors.

Dependencies:
+ [PyTorch](https://github.com/pytorch/pytorch) (version=1.30 is used)
+ [h5py](https://www.h5py.org/)
+ [Faiss](https://github.com/facebookresearch/faiss)
+ [scikit-learn](https://scikit-learn.org/)
+ [NumPy](https://numpy.org/)
+ [tensorboardX](https://github.com/lanpa/tensorboardX)
+ [PyYAML](https://pyyaml.org/)
