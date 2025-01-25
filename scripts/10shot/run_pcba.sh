dataset="pcba"
for seed in $(seq 0 9)
    do
    echo "dataset: $dataset, seed: $seed"
    python run.py --dataset $dataset --random_seed $seed --n_support 10 --gpu 2 --mol_pretrain_load_path ./pretrained_encoders/supervised_contextpred.pth --norm_w 0.1
    done