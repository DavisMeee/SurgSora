export CUDA_VISIBLE_DEVICES=2,3
export TORCH_DISTRIBUTED_DEBUG=DETAIL

EXP_NAME="train_stage1"

accelerate launch --main_process_port 29500 train_stage1_rgbd.py \
 --pretrained_model_name_or_path="./Training/ckpts/stable-video-diffusion-img2vid-xt-1-1"\
 --output_dir="logs/${EXP_NAME}/" \
 --width=256 \
 --height=256 \
 --seed=42 \
 --learning_rate=2e-5 \
 --per_gpu_batch_size=2 \
 --num_train_epochs=20 \
 --max_train_steps=3010 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=1 \
 --checkpointing_steps=1000 \
 --checkpoints_total_limit=100 \
 --validation_steps=1000 \
 --num_frames=21 \
 --gradient_checkpointing \
 --num_validation_images=4 \
 --sample_stride=1 \