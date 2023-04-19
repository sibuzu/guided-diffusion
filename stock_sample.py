"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import io
from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.stock_datasets import load_data
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    with os.open(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

def plot_chart(fname, ary, days, title):
    df = pd.DataFrame(ary)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df['Volume'] = df['Volume'] + 1
    if days is None:
        df['Date'] = pd.date_range(start='2021-01-01', periods=df.shape[0])
    else:
        refdate = pd.to_datetime('1970-01-01')
        dlst = [refdate + pd.Timedelta(days=n) for n in days]
        df['Date'] = dlst
        title = f"{title} ({dlst[0]:%Y-%m-%d})"

    df = df.set_index('Date', drop=True)
    fig, ax = mpf.plot(df, volume=True, returnfig=True, datetime_format='%Y-%m-%d')
    ax[0].set_title(title)

    fig.savefig(fname)
    plt.close(fig)

def save_charts(xpath, prefix, samples, xdays, flist, xbase):
    if not os.path.exists(xpath):
        os.makedirs(xpath, exist_ok=True)

    for x in range(samples.shape[0]):
        fname = f"{xpath}/{prefix}_{xbase+x:06d}.png"
        ary = samples[x,:,:].cpu().numpy()
        ary = np.swapaxes(ary, 0, -1)
        days = xdays[x, :].tolist() if xdays is not None else None
        title = "Generated"
        if flist is not None:
            title = flist[x]
            title = os.path.basename(title).split('.')[0]
        plot_chart(fname, ary, days, title)

def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist()
    logger.configure(args.log_dir)

    logger.log("get data from original...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        stock_size=args.stock_size,
        quick_sampling=False,
    )
    for sample, xdays, flist in data:
        save_charts(args.output_dir, "norm", sample, xdays, flist, 0)
        break

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print(f"model_path = {args.model_path}")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path)
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log(f"sampling {args.num_samples} ...")
    all_images = []
    all_labels = []
    xbase = 0
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 5, args.stock_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        # save chart
        save_charts(args.output_dir, "gene", sample, None, None, xbase)
        xbase += sample.shape[0]

        # concat all_images
        sample = sample.permute(0, 2, 1)
        sample = sample.contiguous()
        all_images.extend([sample.cpu().numpy()])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(args.output_dir, f"gene_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr)

    # dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=10,
        use_ddim=False,
        model_path="./train_log/ema_0.9999_020000.pt",
        data_dir="./data/TWStock",
        log_dir="./sample_log",
        output_dir="./stock_output",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
