# NER_Model
The training script for NER requires the seqeval package:
 pip install seqeval --user
 pip install mxnet


 python finetune_ernie.py \
    --train-path ${DATA_DIR}/train.txt \
    --dev-path ${DATA_DIR}/dev.txt \
    --test-path ${DATA_DIR}/test.txt \
    --gpu 0 --learning-rate 1e-5 --dropout-prob 0.1 --num-epochs 100 --batch-size 8 \
    --optimizer ernieadam --ernie-model ernie_12_768_12 \
    --save-checkpoint-prefix ${MODEL_DIR}/large_bert --seed 13531
