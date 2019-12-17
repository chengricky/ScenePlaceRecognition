#Train
# python AttentionRetrieval.py \
# --dataset=pittsburgh \
# --mode=train \
# --resume=checkpoints_mbnet_pitts30_n14/ \
# --start-epoch=2 \
# --arch=mobilenet2 \
# --numTrain=14 \
# --weightDecay=0.001 \
# --cacheBatchSize=96 \
# --batchSize=3 \
# --threads=4

# TEST
python AttentionRetrieval.py \
--dataset=pittsburgh \
--mode=test \
--split=test \
--ckpt=best \
--resume=checkpoints_mbnet_pitts30_n11/ \
--arch=mobilenet2 \
--cacheBatchSize=96 \
--threads=4
