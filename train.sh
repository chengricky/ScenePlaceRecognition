python AttentionRetrieval.py \
--dataset=pittsburgh \
--mode=test \
--savePath=ckpt_vgg16_pitts30_aug15_a \
--resume=ckpt_vgg16_pitts30_aug15_a \
--start-epoch=6 \
--arch=vgg16 \
--numTrain=5 \
--weightDecay=0.001 \
--cacheBatchSize=64 \
--batchSize=4 \
--withAttention \
--threads=4
