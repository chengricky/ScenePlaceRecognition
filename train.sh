python AttentionRetrieval.py \
--dataset=pittsburgh \
--mode=train \
--savePath=ckpt_vgg16_pitts30_aug5_a \
--arch=vgg16 \
--numTrain=5 \
--weightDecay=0.01 \
--cacheBatchSize=72 \
--batchSize=5 \
--withAttention \
--vladv2 \
--threads=4
