CUDA_VISIBLE_DEVICES=0 python mc.py \
--save_dir="ckpt/vl/frozen/debertav3/intent_trpo_feature" \
--lr=5e-5 --schedule=linear_with_warmup \
--suffix="." --batch_size=32 \
--model_name "ckpt/deberta-v3-base" \
--batch_size_val=32 \
--max_tokens=256 \
--epochs=10 \
--load "best_model.pth" \
--head False 

# bert
# deberta