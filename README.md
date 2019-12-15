# The Unified Network for Scene Description and Scene Classification

The code achieves the unified network with two branches for scene classification and scene description (NetVLAD). The two brahches of the network are trained separately on the classification dataset (Places-365) and the description dataset (Pittsburgh). The traied networks are combined together to test on multiple real-world datasets.

If you use this code, please refer to the paper:
```
Unifying Visual Localization and Scene Recognition for Assistive Navigation
```

The directory structure of this repository is:
```
|--ScenePlaceRecognition
|----DataSet
|----Place365
|----UnifiedModel
```

In this code, the backbones (Wide ResNet-18, MobileNet V2 and ShuffleNet V2) are supported in the unified network. 

The usage of the training and testing code.

`AttentionRetrieval.py` trains the NetVLAD network (with attention module). The affiliated files used in this script is
+ `arguments.py`: read the arguments of the command
+ `DataSet/loadDataset.py`: get DataSet & DataLoader of the designated train, validation or test dataset
+ `NetAVLAD/model.py`: return Net-A-VLAD network
+ `loadCkpt.py`: load checkpoint to the model
+ `DimReduction.py`: train the dimension reduction layer (PCA with whitening) based on NetVLAD. 

`Place365/train_PlacesCNN.py` trains the scene classification network.

`ScenePlaceRecognitionMain.py` tests the unified network on different datasets, meanwhile saves the results of scene classification and scene descriptors.
