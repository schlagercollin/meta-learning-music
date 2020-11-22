#! /bin/bash

# for embed_dim in 64 128 256
# do
#     for hidden_dim in 64 128 256
#     do 
#         echo "Embed = ${embed_dim}"
# 		echo "Hidden = ${hidden_dim}"
# 		echo "==================================="
# 		python train_baseline.py \
# 			--experiment_name maestro_transformer_baseline_seq120_embed${embed_dim}_hidden${hidden_dim} \
# 			--embed_dim ${embed_dim} \
# 			--hidden_dim ${hidden_dim} \
# 			--num_blocks 3 \
# 			--model_type SimpleTransformer \
# 			--context_len 120 \
# 			--dataset maestro \
# 			--num_epochs 20
# 		echo ""
#     done
# done

python train_baseline.py \
	--experiment_name lahk_transformer_baseline_seq240 \
	--embed_dim 128 \
	--hidden_dim 128 \
	--num_blocks 3 \
	--model_type SimpleTransformer \
	--context_len 240 \
	--dataset lakh \
	--num_epochs 20
