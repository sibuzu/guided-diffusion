set DATADIR="./data/TWStock/train"

set MODEL=./tw_log/ema_0.9999_040000.pt
set OUTPUT=./stock_output/ema_040000
REM python stock_sample.py --data_dir %DATADIR% --stock_size 64 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --timestep_respacing 1000 --noise_schedule linear --num_samples 1000 --num_charts 100 --batch_size 100 --model_path %MODEL% --output_dir %OUTPUT%

set DATADIR=""
set MODEL=./tw_log/model040000.pt
set OUTPUT=./stock_output/mdl_040000
python stock_sample.py --data_dir %DATADIR% --stock_size 64 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --timestep_respacing 1000 --noise_schedule linear --num_samples 1000 --num_charts 100 --batch_size 100 --model_path %MODEL% --output_dir %OUTPUT%

set MODEL=./tw_log/ema_0.9999_030000.pt
set OUTPUT=./stock_output/ema_030000
python stock_sample.py --data_dir %DATADIR% --stock_size 64 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --timestep_respacing 1000 --noise_schedule linear --num_samples 1000 --num_charts 100 --batch_size 100 --model_path %MODEL% --output_dir %OUTPUT%

set MODEL=./tw_log/model030000.pt
set OUTPUT=./stock_output/mdl_030000
python stock_sample.py --data_dir %DATADIR% --stock_size 64 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --timestep_respacing 1000 --noise_schedule linear --num_samples 1000 --num_charts 100 --batch_size 100 --model_path %MODEL% --output_dir %OUTPUT%

set MODEL=./tw_log/ema_0.9999_020000.pt
set OUTPUT=./stock_output/ema_020000
python stock_sample.py --data_dir %DATADIR% --stock_size 64 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --timestep_respacing 1000 --noise_schedule linear --num_samples 1000 --num_charts 100 --batch_size 100 --model_path %MODEL% --output_dir %OUTPUT%

set MODEL=./tw_log/model020000.pt
set OUTPUT=./stock_output/mdl_020000
python stock_sample.py --data_dir %DATADIR% --stock_size 64 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --timestep_respacing 1000 --noise_schedule linear --num_samples 1000 --num_charts 100 --batch_size 100 --model_path %MODEL% --output_dir %OUTPUT%

set MODEL=./tw_log/ema_0.9999_010000.pt
set OUTPUT=./stock_output/ema_010000
python stock_sample.py --data_dir %DATADIR% --stock_size 64 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --timestep_respacing 1000 --noise_schedule linear --num_samples 1000 --num_charts 100 --batch_size 100 --model_path %MODEL% --output_dir %OUTPUT%

set MODEL=./tw_log/model010000.pt
set OUTPUT=./stock_output/mdl_010000
python stock_sample.py --data_dir %DATADIR% --stock_size 64 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --timestep_respacing 1000 --noise_schedule linear --num_samples 1000 --num_charts 100 --batch_size 100 --model_path %MODEL% --output_dir %OUTPUT%
