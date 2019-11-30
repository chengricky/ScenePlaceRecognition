python AttentionRetrieval.py \
--dataset=pittsburgh \
--mode=train \
--saveDecs \
--withAttention \
--vladv2 \
--arch=mobilenet \
--numTrain=5 \
--weightDecay=0.001 \
--cacheBatchSize=32 \
--batchSize=4 \
--threads=4

