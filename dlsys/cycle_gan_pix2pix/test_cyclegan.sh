#!/bin/bash

dataset=monet2photo

python3 test.py \
        --dataroot "./datasets/${dataset}" \
        --name "${dataset}_cyclegan" \
        --model cycle_gan \
        --phase test \
        --no_dropout
