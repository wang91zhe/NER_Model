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
    --save-checkpoint-prefix ${MODEL_DIR}/large_ernie --seed 13531

# 模型结构
----ernie\
   |----predict_data\
   |----save_model\
   |----.idea\
   |----server_ernie\
   |----data\
   |----.git\
   |----label_map\
   |----pretrained\
   |----ERNIE\
   |----model_dir\
   |----logger_file\

predict_data：存放预测的文件
save_model：模型文件
server_ernie：启动服务，启动Server，只需启动app_server.py文件即可
label_map：每个HSCODE对应的标签文件
pretrained：Ernie的预训练模型文件及参数文件
ERNIE：百度的ernie模型
model_dir：每个HSCODE模型路径文件
logger_file：日志文件
 
 

