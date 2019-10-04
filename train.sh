python AttentionRetrieval.py \
--dataset=pittsburgh \
--mode=train \
--savePath=ckpt_res18_pitts30_aug_a \
--withAttention \
--arch=resnet18 \
--numTrain=5 \
--weightDecay=0.01 \
--cacheBatchSize=24
