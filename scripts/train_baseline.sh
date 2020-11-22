#! /bin/bash

python train_baseline.py \
	--experiment_name maestro_transformer_baseline_seq240_blocks5 \
	--num_blocks 5 \
	--model_type SimpleTransformer \
	--context_len 240 \
	--dataset maestro \
	--num_epochs 30
