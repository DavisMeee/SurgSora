export CUDA_VISIBLE_DEVICES=2,3
EXP_NAME="train_stage2"

accelerate launch --main_process_port 29500 train_stage2_rgbd_traj.py \
 --pretrained_model_name_or_path="./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1" \
 --controlnet_model_name_or_path="./Training/logs/train_stage1/checkpoint-3000/controlnet" \
 --output_dir="./logs/${EXP_NAME}/" \
 --height=256 \
 --width=256 \
 --train_height=256 \
 --train_width=256 \
 --seed=42 \
 --learning_rate=2e-5 \
 --per_gpu_batch_size=2 \
 --num_train_epochs=20 \
 --max_train_steps=50000 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=1 \
 --checkpointing_steps=1000 \
 --checkpoints_total_limit=100 \
 --validation_steps=1000 \
 --gradient_checkpointing \
 --num_validation_images=4 \
 --use_8bit_adam \
 --sample_stride=4 \
 --num_frames=21 \
