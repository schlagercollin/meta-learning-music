#! /bin/bash

# Lakh
# python -m sampling.sample_baseline \
# 	--experiment_name TransformerBaseline3Blocks \
# 	--model_type SimpleTransformer \
# 	--dataset lakh \
#     --load_from_iteration 37000 \
#     --num_batch_gen 128

# Mestro
python -m sampling.sample_baseline \
	--experiment_name maestro_transformer_baseline_seq120_embed128_hidden128 \
	--model_type SimpleTransformer \
	--dataset maestro \
    --load_from_iteration 56000 \
    --num_batch_gen 128