# Place Recognition Besed on Net-A-VLAD

The code achieves NetVLAD with attention mechanism training and testing on multiple datasets.

The usage of the training and testing code.

`AttentionRetrieval.py` trains the NetVLAD network with attention module. The affiliated files used in this script is
+ `arguments.py`: read the arguments of the command
+ `DataSet/loadDataset.py`: get DataSet & DataLoader of the designated train, validation or test dataset
+ `NetAVLAD/model.py`: return Net-A-VLAD network
+ `loadCkpt.py`: load checkpoint to the model

## Attention Module on Tokyo Dataset

The attention module is 