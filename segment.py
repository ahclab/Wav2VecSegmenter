import gc
import itertools
import os
import subprocess
import sys
from multiprocessing import cpu_count
from pathlib import Path
from shlex import split

import hydra
import logzero
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lib.dataset import FixedSegmentationDatasetNoTarget
from lib.datautils import AutoRegCollateFn, CollateFn
from lib.evaluate import infer
from lib.segment import pdac, pdac_with_logits, pthr, strm, update_yaml_content
from logzero import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


def generate(config: DictConfig) -> list:
    device = (
        torch.device(f"cuda:0")
        if torch.cuda.device_count() > 0
        else torch.device("cpu")
    )

    # load vocabulary
    if config.task.vocab:
        vocab = instantiate(config.task.vocab)
        config.task.model["vocab_size"] = vocab.vocab_size
    else:
        vocab = None

    # build model
    model = instantiate(
        config.task.model,
    ).to(device)

    # load checkpoint
    checkpoint = torch.load(config.ckpt_path, map_location=device)

    if config.task.model.finetune_wav2vec:
        model.load_state_dict(checkpoint["state_dict"])
    else:  # do not store wav2vec2 state_dict
        model.seg_model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # load settings
    autoregression = config.task.autoregression
    loss_tag = config.task.loss.tag

    # collate_fn
    if not autoregression and not vocab:
        segm_collate_fn = CollateFn(pad_token_id=0)
    elif not autoregression:
        segm_collate_fn = CollateFn(pad_token_id=vocab.pad_token_id)
    else:
        segm_collate_fn = AutoRegCollateFn(pad_token_id=vocab.pad_token_id)

    wav_dir = config.infer_data.wav_dir
    with open(config.infer_data.orig_seg_yaml, "r") as f:
        seg_yaml = yaml.load(f, Loader=yaml.SafeLoader)

    yaml_content = []
    for wav_path, group in tqdm(itertools.groupby(seg_yaml, key=lambda x: x["wav"])):
        wav_path = Path(wav_dir) / wav_path

        # initialize a dataset for the fixed segmentation
        dataset = FixedSegmentationDatasetNoTarget(
            wav_path, config.inference_segment_length, config.inference_times
        )
        sgm_frame_probs = None

        for inference_iteration in range(config.inference_times):
            # create a dataloader for this fixed-length segmentation of the wav file
            dataset.fixed_length_segmentation(inference_iteration)
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=min(cpu_count() // 2, 4),
                shuffle=False,
                drop_last=False,
                collate_fn=segm_collate_fn,
            )

            # get frame segmentation frame probabilities in the output space
            probs, logits, _, _ = infer(
                model,
                dataloader,
                device,
                autoregression,
                loss_tag,
                vocab,
            )
            if sgm_frame_probs is None:
                sgm_frame_probs = probs.copy()
                sgm_frame_logits = logits.copy()
            else:
                sgm_frame_probs += probs
                sgm_frame_logits += logits

        sgm_frame_probs /= config.inference_times

        algo_conf = OmegaConf.to_object(config.algorithm)
        algorithm = algo_conf.pop("tag")

        if algorithm == "dac":
            segments = pdac(sgm_frame_probs, **algo_conf)
        elif algorithm == "dac_logits":
            segments = pdac_with_logits(
                sgm_frame_probs, sgm_frame_logits, vocab, **algo_conf
            )
        elif algorithm == "strm":
            segments = strm(sgm_frame_probs, **algo_conf)
        elif algorithm == "pthr":
            segments = pthr(sgm_frame_probs, **algo_conf)

        yaml_content = update_yaml_content(yaml_content, segments, wav_path.name)

    del model
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()

    return yaml_content


def init(config: DictConfig) -> None:
    logzero.logfile("log")

    # logging
    logger.info(f"Host: [{os.uname()[1]}]")
    logger.info(f'CUDA_VISIBLE_DEVICES = {os.environ.get("CUDA_VISIBLE_DEVICES", "")}')

    git_check = subprocess.run(
        split("git rev-parse --is-inside-work-tree"), capture_output=True, check=False
    )
    if git_check.returncode == 0:
        git_log = subprocess.check_output(split("git log -n1"), encoding="utf-8")
        git_diff = subprocess.check_output(split("git diff"), encoding="utf-8")
        logger.info(
            "Git repository is found. Dumpling logs & diffs...\n"
            f"{git_log}\n{git_diff}"
        )
    else:
        logger.info("Git repository is not found.")

    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Command is executed at: [{config.work_dir}]")
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")


@hydra.main(config_path="conf", config_name="segment")
def main(config: DictConfig) -> None:
    if config.config_path is not None:
        prev_cfg = OmegaConf.load(config.config_path)
        config = OmegaConf.merge(prev_cfg, config)

    init(config)

    result_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    logger.info(f"Output directory : [{result_dir}]")
    cust_seg_yaml = Path(result_dir) / config.cust_seg_yaml

    yaml_content = generate(config)

    logger.info(f"Number of segments: {len(yaml_content)}")

    with open(cust_seg_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=True)
    logger.info(f"Saved to [{cust_seg_yaml}].")


if __name__ == "__main__":
    main()
