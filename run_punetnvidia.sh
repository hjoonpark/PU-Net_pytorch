clear;

gpu=0
model=punet
extra_tag=punet_baseline

# nohup python -u train.py \
#     --model ${model} \
#     --batch_size 32 \
#     --log_dir logs/${extra_tag} \
#     --gpu ${gpu} \
#     >> logs/${extra_tag}/nohup.log 2>&1 &
mkdir output
mkdir output/${extra_tag}
python -u train_nvidia.py \
    --model ${model} \
    --log_dir output/${extra_tag} \
    --gpu ${gpu}
    
