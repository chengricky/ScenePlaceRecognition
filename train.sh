python AttentionRetrieval.py \
--dataset=pittsburgh \
--mode=train \
--savePath=checkpoints_sfnet_pitts30_n5/ \
--arch=shufflenet2 \
--numTrain=5 \
--weightDecay=0.001 \
--cacheBatchSize=48 \
--batchSize=1 \
--threads=4

