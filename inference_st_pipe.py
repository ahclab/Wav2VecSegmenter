import os
import subprocess
import sys
from multiprocessing import cpu_count
from pathlib import Path
from shlex import split

import hydra
import logzero
import torch
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from inference import generate
from lib.dataset import FixedSegmentationDatasetNoTarget
from lib.datautils import AutoRegCollateFn, CollateFn
from lib.eval_scripts.format_generation_output import format_generation_output
from lib.eval_scripts.original_segmentation_to_xml import original_segmentation_to_xml
from lib.eval_scripts.prepare_custom_dataset import prepare_custom_dataset
from lib.eval_scripts.score import score_bertscore, score_bleurt, score_sacrebleu
from logzero import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


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


@hydra.main(config_path="conf", config_name="inference")
def main(config: DictConfig) -> None:
    if config.base_cfg is not None:
        prev_cfg = OmegaConf.load(Path(config.base_cfg) / "config.yaml")
        config = OmegaConf.merge(prev_cfg, config)

    init(config)

    results_path = Path(os.getcwd())
    cust_seg_yaml = results_path / config.cust_seg_yaml

    # add a new key to OmegaConf obj for logging
    OmegaConf.set_struct(config, False)
    config["results_path"] = str(results_path)

    wandb_name = "/".join([config.exp_name, results_path.name])
    if config.log_wandb:
        run = wandb.init(
            project=config.project_name,
            config=config,
            group=config.group,
            name=wandb_name,
            tags=config.tags,
            notes=config.notes,
            dir=str(results_path),
        )
    wandb_dict = dict()

    # segment audio
    yaml_content = generate(config)
    with open(cust_seg_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=True)
    wandb_dict["n_segments"] = len(yaml_content)

    # prepare the tsv file from the custom segmentation yaml
    prepare_custom_dataset(
        cust_seg_yaml,
        config.infer_data.wav_dir,
        config.infer_data.tgt_lang,
        0,
    )

    # translate using the Speech Trasnlation model
    if os.path.basename(config.st_model_dir) == "joint-s2t-mustc-en-de":
        fairseq_generate_cmd = (
            f"fairseq-generate {results_path}"
            " --task speech_text_joint_to_text"
            " --max-tokens 100000"
            " --max-source-positions 12000"
            " --nbest 1"
            " --batch-size 128"
            f" --path {config.st_model_dir}/{config.st_ckpt}"
            f" --gen-subset {Path(config.cust_seg_yaml).stem}"
            f" --config-yaml {config.st_model_dir}/config.yaml"
            f" --beam 5"
            " --lenpen 1.0"
            f" --user-dir {config.fairseq_root}/examples/speech_text_joint_to_text"
            f" --load-speech-only > {results_path}/translations.txt"
        )
    elif os.path.basename(config.st_model_dir) == "mustc_multilingual_st":
        fairseq_generate_cmd = (
            f"fairseq-generate {results_path}"
            " --task speech_to_text"
            f" --path {config.st_model_dir}/{config.st_ckpt}"
            f" --gen-subset {Path(config.cust_seg_yaml).stem}"
            f" --config-yaml {config.st_model_dir}/config.yaml"
            " --max-tokens 50000"
            " --beam 5"
            f" --prefix-size 1 > {results_path}/translations.txt"
        )
    else:
        raise ValueError("Unknown model dir")
    subprocess.run(fairseq_generate_cmd, shell=True)

    # extract raw hypotheses from fairseq-generate output
    format_generation_output(f"{results_path}/translations.txt")

    original_segmentation_to_xml(
        config.infer_data.orig_seg_yaml,
        config.infer_data.orig_src_txt,
        config.infer_data.orig_tgt_txt,
        results_path,
    )

    # align the hypotheses with the references
    split_name = Path(config.infer_data.orig_seg_yaml).stem
    sysid = Path(config.st_model_dir).stem
    mwersegment_cmd = (
        f"{config.mwersegmenter_root}/segmentBasedOnMWER.sh"
        f" {results_path}/{split_name}.{config.infer_data.src_lang}.xml"
        f" {results_path}/{split_name}.{config.infer_data.tgt_lang}.xml"
        f" {results_path}/translations_formatted.txt"
        f" {sysid} {config.infer_data.tgt_lang}"
        f" {results_path}/translations_aligned.xml normalize 1"
    )
    subprocess.run(mwersegment_cmd, shell=True)

    # obtain scores of the aligned hypothesis and references
    if "bleu" in config.st_metrics:
        from sacrebleu.metrics.bleu import BLEUScore

        bleu = score_sacrebleu(
            f"{results_path}/__mreference",
            f"{results_path}/__segments",
        )
        bleu_print = str(bleu)
        bleu = bleu.score
        with open(f"{results_path}/score.sacrebleu", "w") as f:
            f.write(bleu_print)
        wandb_dict["bleu"] = bleu

        columns = ["name", "print", "score"]
        table = [
            [wandb_name, bleu_print, bleu],
        ]
        wandb_table = wandb.Table(data=table, columns=columns)
        wandb_dict["bleu_table"] = wandb_table

    if "bertscore" in config.st_metrics:
        bertscore = score_bertscore(
            f"{results_path}/__mreference",
            f"{results_path}/__segments",
            f"{config.infer_data.tgt_lang}",
        )
        p = bertscore[0]
        r = bertscore[1]
        f1 = bertscore[2]
        bertscore_print = "BERTScore (P/R/F1) = " f"{p:.4f}/{r:.4f}/{f1:.4f}"
        with open(f"{results_path}/score.bertscore", "w") as f:
            f.write(bertscore_print)
        wandb_dict["bertscore_p"] = p
        wandb_dict["bertscore_r"] = r
        wandb_dict["bertscore_f1"] = f1

        columns = ["name", "print", "p", "r", "f1"]
        table = [
            [wandb_name, bertscore_print, p, r, f1],
        ]
        wandb_table = wandb.Table(data=table, columns=columns)
        wandb_dict["bertscore_table"] = wandb_table

    if "bleurt" in config.st_metrics:
        bleurt = score_bleurt(
            f"{results_path}/__mreference",
            f"{results_path}/__segments",
            f"{config.bleurt_path}",
        )
        bleurt_print = f"BLEURT (Average) = {bleurt:.4f}"
        with open(f"{results_path}/score.bleurt", "w") as f:
            f.write(bleurt_print)
        wandb_dict["bleurt"] = bleurt

        columns = ["name", "print", "score"]
        table = [
            [wandb_name, bleurt_print, bleurt],
        ]
        wandb_table = wandb.Table(data=table, columns=columns)
        wandb_dict["bleurt_table"] = wandb_table

    if config.log_wandb:
        wandb.log(wandb_dict, step=0)
        wandb.finish()


if __name__ == "__main__":
    main()
