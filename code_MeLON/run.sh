#!/bin/sh -x

python main.py --model_name NCF --gpu '0' --epoch 100 --l2 1e-6 --dataset 'Yelp' --meta_weighting 1  --train_ratio 0.95  --meta_name MeLON --max_edge 10 &
