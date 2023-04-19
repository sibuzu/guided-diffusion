set DATADIR=".\data\TWStock\train"

python stock_train.py --data_dir %DATADIR% --stock_size 64 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --noise_schedule linear --lr 1e-4 --batch_size 1024 --log_dir ./tw_log

