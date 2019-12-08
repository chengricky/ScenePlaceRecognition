python AttentionRetrieval.py \
--dataset=pittsburgh \
--mode=train \
--savePath=checkpoints_mbnet_pitts30_n5/ \
--vladv2 \
--arch=mobilenet \
--numTrain=5 \
--weightDecay=0.001 \
--cacheBatchSize=104 \
--batchSize=5 \
--threads=4

