set DATADIR="./data/TWStock/train"
set MODEL=./train_log/ema_0.9999_040000.pt
set OUTPUT=./stock_output/test

python stock_sample.py --data_dir %DATADIR% --stock_size 64 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --timestep_respacing 100 --noise_schedule linear --num_samples 20 --num_charts 12 --batch_size 10 --model_path %MODEL% --output_dir %OUTPUT%
