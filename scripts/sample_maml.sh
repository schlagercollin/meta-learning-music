#! /bin/bash

# Lakh
# python -m sampling.sample_maml \
#     --experiment_name TransformerMAML3Blocks \
#     --model_type SimpleTransformer \
#     --dataset lakh \
#     --load_from_iteration 13308 \
#     --repeat_generation 50 \

# Maestro
python -m sampling.sample_maml \
    --experiment_name MaestroTransformerMAML_final \
    --model_type SimpleTransformer \
    --dataset maestro \
    --load_from_iteration 2000 \
    --repeat_generation 50 \