dataset="sider"
for seed in $(seq 0 9)
    do
    echo "dataset: $dataset, seed: $seed"
    python run.py --dataset $dataset --random_seed $seed --n_support 5 --gpu 1 --mol_pretrain_load_path ./pretrained_encoders/supervised_contextpred.pth
    done