"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
# import torch.distributed as dist
import blobfile as bf
import io
from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

def save_samples(xpath, samples):
    if not os.path.exists(xpath):
        os.makedirs(xpath, exist_ok=True)
    files = os.listdir(xpath)
    last_index = -1
    for fname in files:
        try:
            if fname[-4:] != ".png":
                continue
            index = int(fname[:-4])
            if index > last_index:
                last_index = index
        except:
            pass
    
    for x in range(samples.shape[0]):
        last_index += 1
        fname = f"{xpath}/{last_index:06d}.png"

        # save images
        ary = samples[x,:,:,:].cpu().numpy()
        # print(f"{ary[:].max()}, {ary[:].min()}, {ary.dtype}")
        # print(f"ary_shape={ary.shape}")
        image = Image.fromarray(ary)
        image.save(fname)


def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist()
    logger.configure(args.log_dir)

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
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        save_samples(args.output_dir, sample)

        # gathered_samples = th.zeros_like(sample)
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy()])
        if args.class_cond:
            gathered_labels = th.zeros_like(classes) 
            # dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    if args.class_cond:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)

    # dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=64,
        batch_size=8,
        use_ddim=False,
        model_path="",
        log_dir="./sample_log",
        output_dir="./output",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
