CUDA_VISIBLE_DEVICES=0 python \
flamingo.py \
--model_name flamingo \
--cnn_head false \
--seed 42 \
--do_train \
--do_eval \
--per_device_eval_batch_size 16 \
--logging_steps 500 \
--report_to tensorboard \
--output_dir ckpt/vl/flamingo/intent \
--overwrite_output_dir \
--dataset_path  dataset_trpo/intent \
--dataloader_num_workers 16 \
--task_name intent \
--pretrained True \

# survival_trpo
# survival_obs
# goal_trpo
# goal_obs