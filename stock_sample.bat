set MODEL=./train_log/ema_0.9999_020000.pt
set OUTPUT=./output/ema_020000

python stock_sample.py --image_size 64 --num_channels 128 --num_res_blocks 3 --diffusion_steps 1000 --timestep_respacing 250 --noise_schedule linear --num_samples 10 --batch_size 10 --model_path %MODEL% --output_dir %OUTPUT%
