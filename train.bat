set DATADIR=".\data\imagenet_64\train_64x64"

python image_train.py --data_dir %DATADIR% --image_size 64 --num_channels 128 --num_res_blocks 3 --diffusion_steps 100 --noise_schedule linear --lr 1e-4 --batch_size 4 

