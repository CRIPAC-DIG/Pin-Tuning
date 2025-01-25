dataset="toxcast-CEETOX"
for seed in $(seq 0 9)
    do
    echo "dataset: $dataset, seed: $seed"
    python run.py --dataset $dataset --random_seed $seed --n_support 10 --gpu 3 --mol_pretrain_load_path ./pretrained_encoders/supervised_contextpred.pth
    done