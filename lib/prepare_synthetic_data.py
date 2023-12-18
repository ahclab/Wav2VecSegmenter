import argparse
import gc
import math
import os
import subprocess
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path

import sacrebleu
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from scipy.stats.mstats import gmean
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import FixedSegmentationDatasetNoTarget
from datautils import CollateFn
from eval_scripts.format_generation_output import format_generation_output
from eval_scripts.original_segmentation_to_xml import original_segmentation_to_xml
from eval_scripts.prepare_custom_dataset import prepare_custom_dataset
from eval_scripts.score import score_bertscore, score_sacrebleu
from evaluate import infer
from segment import pdac_tree, update_tree_yaml_content


def generate_segmentation_tree(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_config = OmegaConf.load(Path(args.outputs) / ".hydra/config.yaml")

    # 1. load model
    device = (
        torch.device(f"cuda:0")
        if torch.cuda.device_count() > 0
        else torch.device("cpu")
    )

    model = instantiate(
        train_config.task.model,
    ).to(device)

    # load checkpoint
    checkpoint = torch.load(
        "/".join([args.outputs, train_config.exp_name, "ckpts", args.checkpoint]),
        map_location=device,
    )

    if train_config.task.model.finetune_wav2vec:
        model.load_state_dict(checkpoint["state_dict"])
    else:  # do not store wav2vec2 state_dict
        model.seg_model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # load settings
    autoregression = train_config.task.autoregression
    loss_tag = train_config.task.loss.tag
    segm_collate_fn = CollateFn(pad_token_id=0)

    # 2. generate tree
    segmentation_tree_yaml = Path(save_dir / "custom_segments.tree.yaml")

    yaml_content = []
    for wav_path in sorted(list(Path(args.path_to_wavs).glob("*.wav"))):
        # initialize a dataset for the fixed segmentation
        dataset = FixedSegmentationDatasetNoTarget(
            wav_path,
            args.inference_segment_length,
            args.inference_times,
        )
        sgm_frame_probs = None

        for inference_iteration in range(args.inference_times):
            # create a dataloader for this fixed-length segmentation of the wav file
            dataset.fixed_length_segmentation(inference_iteration)
            dataloader = DataLoader(
                dataset,
                batch_size=args.inference_batch_size,
                num_workers=min(cpu_count() // 2, 4),
                shuffle=False,
                drop_last=False,
                collate_fn=segm_collate_fn,
            )

            # get frame segmentation frame probabilities in the output space
            probs, _, _ = infer(
                model,
                dataloader,
                device,
                autoregression,
                loss_tag,
                vocab=None,
            )
            if sgm_frame_probs is None:
                sgm_frame_probs = probs.copy()
            else:
                sgm_frame_probs += probs

        sgm_frame_probs /= args.inference_times

        # apply the probabilistic dac to the segmentation frame probabilities
        tree = pdac_tree(
            sgm_frame_probs,
            args.max_segment_length,
            args.min_segment_length,
            args.boundary_threshold,
            args.trim_threshold,
            args.tree_depth,
        )
        with open(save_dir / "tree.length", mode="a") as f:
            f.write(f"{wav_path.name}\t{str(len(tree))}\n")
        yaml_content = update_tree_yaml_content(
            yaml_content,
            tree,
            wav_path.name,
            args.max_segment_length,
            args.min_segment_length,
        )

    del model
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()

    # 3. dump yaml file & prepare dataset for Fairseq ST
    with open(segmentation_tree_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=True)


def generate_translation_tree(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    segmentation_tree_yaml = Path(save_dir / "custom_segments.tree.yaml")

    prepare_custom_dataset(
        segmentation_tree_yaml,
        args.path_to_wavs,
        args.tgt_lang,
        0,
        sort_by_offset=False,
    )

    # translate using the Speech Trasnlation model
    fairseq_generate_cmd = (
        f"fairseq-generate {save_dir}"
        " --task speech_text_joint_to_text"
        " --max-tokens 100000"
        " --max-source-positions 12000"
        " --nbest 1"
        " --batch-size 128"
        f" --path {args.path_to_st_checkpoint}"
        f" --gen-subset {segmentation_tree_yaml.stem}"
        f" --config-yaml {Path(args.path_to_st_checkpoint).parent}/config.yaml"
        " --beam 5"
        " --lenpen 1.0"
        f" --user-dir {args.fairseq_root}/examples/speech_text_joint_to_text"
        f" --load-speech-only > {save_dir}/translations.txt"
    )
    subprocess.run(fairseq_generate_cmd, shell=True)

    # extract raw hypotheses (translation tree) from fairseq-generate output
    format_generation_output(f"{save_dir}/translations.txt")


def tournament(
    metrics: str,
    depth: int,
    tgt_tree: list[str],
    tgt_segments: list[list[str]],
    src_segments: list[list[dict]],
    ref_talks: list[str],
    path_to_output_segments: str,
    path_to_output_translations_segments: str,
    path_to_output_translations_talks: str,
):
    for l in range(depth, 0, -1):
        for i in range(0, 2**l, 2):
            p_parent = 2 ** (l - 1) + int(i / 2) - 1
            p_child_a = 2**l + i - 1
            p_child_b = 2**l + i

            child_a = tgt_tree[p_child_a]
            child_b = tgt_tree[p_child_b]
            child = " ".join([child_a, child_b]).strip()
            if child == "":
                continue

            parent = tgt_tree[p_parent]
            if parent == "":
                # override parent <- children
                tgt_tree[p_parent] = child

                tgt_segments[p_parent] = (
                    tgt_segments[p_child_a] + tgt_segments[p_child_b]
                )
                src_segments[p_parent] = (
                    src_segments[p_child_a] + src_segments[p_child_b]
                )

                # delete child nodes from tgt and src tree
                tgt_segments[p_child_a] = [""]
                tgt_segments[p_child_b] = [""]
                src_segments[p_child_a] = [{"offset": 10**20}]
                src_segments[p_child_b] = [{"offset": 10**20}]
                continue

            # parent vs children
            if metrics == "BLEU":
                c_score = gmean(sacrebleu.sentence_bleu(child, ref_talks).precisions)
                p_score = gmean(sacrebleu.sentence_bleu(parent, ref_talks).precisions)
            else:
                raise NotImplementedError

            if c_score > p_score:
                # override parent <- children
                tgt_tree[p_parent] = child

                tgt_segments[p_parent] = (
                    tgt_segments[p_child_a] + tgt_segments[p_child_b]
                )
                src_segments[p_parent] = (
                    src_segments[p_child_a] + src_segments[p_child_b]
                )

            # delete child nodes from tgt and src tree
            tgt_segments[p_child_a] = [""]
            tgt_segments[p_child_b] = [""]
            src_segments[p_child_a] = [{"offset": 10**20}]
            src_segments[p_child_b] = [{"offset": 10**20}]

    # output full translation (one line for one document)
    with open(path_to_output_translations_talks, mode="a") as f:
        f.write(tgt_tree[0] + "\n")

    # output segments (src yaml, tgt text)
    with open(path_to_output_segments, mode="a") as fs, open(
        path_to_output_translations_segments, mode="a"
    ) as ft:
        src_segments = src_segments[0]
        tgt_segments = tgt_segments[0]
        idx = [
            i[0]
            for i in sorted(
                enumerate(src_segments), key=lambda x: float(x[1]["offset"])
            )
        ]
        for i in idx:
            if src_segments[i] == {"offset": 10**20}:
                break
            fs.write(f"- {src_segments[i]}\n")
            ft.write(tgt_segments[i] + "\n")


def select_segments(args):
    save_dir = Path(args.save_dir)
    segmentation_tree_yaml = Path(save_dir / "custom_segments.tree.yaml")
    translation_tree_txt = Path(save_dir / "translations_formatted.txt")
    synthetic_dir = save_dir / "synthetic_data"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    tree_lengths = {}
    with open(save_dir / "tree.length", mode="r") as f:
        lines = f.readlines()
    for l in lines:
        wav, length = l.split("\t")
        tree_lengths[wav] = int(length)

    with open(segmentation_tree_yaml, "r") as f:
        segmentation = yaml.load(f, Loader=yaml.BaseLoader)

    with open(translation_tree_txt, "r") as f:
        tgt_lang_text = f.read().splitlines()

    # read data from MuST-C corpus
    with open(args.path_to_src_yaml, "r") as f:
        src_segmentation = yaml.load(f, Loader=yaml.BaseLoader)
    with open(args.path_to_ref_txt, "r") as f:
        ref_texts = f.read().splitlines()

    # load references for each talk
    ref_talks = {}
    txt_pool = []
    curr_wav = src_segmentation[0]["wav"]
    for i, seg in enumerate(src_segmentation):
        if seg["wav"] != curr_wav:
            ref_talks[curr_wav] = [" ".join(txt_pool)]
            curr_wav = seg["wav"]
            txt_pool = [ref_texts[i]]
        else:
            txt_pool.append(ref_texts[i])
    ref_talks[curr_wav] = [" ".join(txt_pool)]

    curr_wav = src_segmentation[0]["wav"]
    tgt_tree = [""] * tree_lengths[curr_wav]
    tgt_segments = [[""] for i in range(tree_lengths[curr_wav])]
    src_segments = [[{"offset": 10**20}] for i in range(tree_lengths[curr_wav])]
    for i, seg in enumerate(segmentation):
        pos = int(seg["speaker_id"])

        if seg["wav"] != curr_wav:
            depth = min(int(math.log2(len(tgt_tree))), args.tree_depth)
            tournament(
                args.metrics,
                depth,
                tgt_tree,
                tgt_segments,
                src_segments,
                ref_talks[curr_wav],
                synthetic_dir / "custom_segments.yaml",
                synthetic_dir / "translations_custom_segments.txt",
                synthetic_dir / "translations_talks.txt",
            )
            print(f"tournament of {curr_wav} is completed")
            curr_wav = seg["wav"]
            tgt_tree = [""] * tree_lengths[curr_wav]
            tgt_segments = [[""] for i in range(tree_lengths[curr_wav])]
            src_segments = [
                [{"offset": 10**20}] for i in range(tree_lengths[curr_wav])
            ]

        tgt_tree[pos] = tgt_lang_text[i]
        tgt_segments[pos] = [tgt_lang_text[i]]
        src_segments[pos] = [seg]

    tournament(
        args.metrics,
        depth,
        tgt_tree,
        tgt_segments,
        src_segments,
        ref_talks[curr_wav],
        synthetic_dir / "custom_segments.yaml",
        synthetic_dir / "translations_custom_segments.txt",
        synthetic_dir / "translations_talks.txt",
    )
    print(f"tournament of {curr_wav} is completed")

    if args.evaluate_data:
        # calculate BLEU score for synhtetic data
        original_segmentation_to_xml(
            args.path_to_src_yaml,
            args.path_to_src_txt,
            args.path_to_ref_txt,
            save_dir,
        )
        split_name = Path(args.path_to_src_yaml).stem
        sysid = Path(args.path_to_st_checkpoint).parent.stem
        mwersegment_cmd = (
            f"{args.mwersegmenter_root}/segmentBasedOnMWER.sh"
            f" {save_dir}/{Path(args.path_to_src_txt).name}.xml"
            f" {save_dir}/{Path(args.path_to_ref_txt).name}.xml"
            f" {synthetic_dir}/translations_custom_segments.txt"
            f" {sysid} {args.tgt_lang}"
            f" {save_dir}/translations_aligned.xml normalize 1"
        )
        subprocess.run(mwersegment_cmd, shell=True)
        bleu = score_sacrebleu(
            f"{synthetic_dir}/__mreference",
            f"{synthetic_dir}/__segments",
        )
        bleu_print = str(bleu)
        with open(f"{synthetic_dir}/score.sacrebleu", "w") as f:
            f.write(bleu_print)

    # save segments as must-c format
    with open(synthetic_dir / "custom_segments.yaml", "r") as f:
        segmentation = yaml.load(f, Loader=yaml.BaseLoader)
    with open(synthetic_dir / "custom_segments.mustc.yaml", "w") as f:
        for seg in segmentation:
            seg = str(seg).replace("'", "")
            f.write(f"- {seg}\n")

    # prepare dataset for sfc training
    sys.path.append(args.shas_root)
    from src.data_prep.prepare_dataset_for_segmentation import (
        prepare_dataset_for_segmentation,
    )

    prepare_dataset_for_segmentation(
        synthetic_dir / "custom_segments.mustc.yaml",
        args.path_to_wavs,
        synthetic_dir,
    )


def main(args):
    print(f"Stage {args.stage}-{args.stop_stage}")
    global_start_time = time.perf_counter()

    stage = args.stage
    while stage <= args.stop_stage:
        if stage == 1:
            print("Stage 1: generate segmentation tree")
            start_time = time.perf_counter()
            generate_segmentation_tree(args)
            end_time = time.perf_counter()
            print(f"Stage 1 finished (Elapsed: {end_time - start_time})")
        elif stage == 2:
            print("Stage 2: generate translation tree")
            start_time = time.perf_counter()
            generate_translation_tree(args)
            end_time = time.perf_counter()
            print(f"Stage 2 finished (Elapsed: {end_time - start_time})")
        elif stage == 3:
            print("Stage 3: select synthetic segments")
            start_time = time.perf_counter()
            select_segments(args)
            end_time = time.perf_counter()
            print(f"Stage 3 finished (Elapsed: {end_time - start_time})")

        stage += 1

    global_end_time = time.perf_counter()
    print(
        (
            f"Stage {args.stage}-{args.stop_stage} finished "
            f"(Elapased: {global_end_time - global_start_time})"
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=int,
        choices=range(1, 4),
        default=1,
    )
    parser.add_argument(
        "--stop_stage",
        type=int,
        choices=range(1, 4),
        default=3,
    )
    parser.add_argument(
        "--outputs",
        type=str,
        required=True,
        help="absolute path to the speech-frame-classifier experiment",
    )
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        type=str,
        required=True,
        help="checkpoint name of the speech-frame-classifier",
    )
    parser.add_argument(
        "--path_to_wavs",
        "-wavs",
        type=str,
        help="absolute path to the directory of the wav audios to be segmented",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="de",
        help="target language",
    )
    parser.add_argument(
        "--inference_batch_size",
        "-bs",
        type=int,
        default=12,
        help="batch size (in examples) of inference with the audio-frame-classifier",
    )
    parser.add_argument(
        "--inference_segment_length",
        "-len",
        type=int,
        default=20,
        help="segment length (in seconds) of fixed-length segmentation during inference"
        "with audio-frame-classifier",
    )
    parser.add_argument(
        "--inference_times",
        "-n",
        type=int,
        default=1,
        help="how many times to apply inference on different fixed-length segmentations"
        "of each wav",
    )
    parser.add_argument(
        "--max_segment_length",
        "-max",
        type=float,
        default=28,
        help="maximum segment length (in seconds)",
    )
    parser.add_argument(
        "--min_segment_length",
        "-min",
        type=float,
        default=8,
        help="minimum segment length (in seconds)",
    )
    parser.add_argument(
        "--boundary_threshold",
        "-bound_thr",
        type=float,
        default=0.5,
        help="outside the segment between the first and last points that are above the threshold can not be a segment boundary",
    )
    parser.add_argument(
        "--trim_threshold",
        "-trim_thr",
        type=float,
        default=0,
        help="after each split by the algorithm, the resulting segments are trimmed to"
        "the first and last points that corresponds to a probability above this value",
    )
    parser.add_argument(
        "--tree_depth",
        "-depth",
        type=int,
        default=20,
        help="upper bound of binary tree depth",
    )
    parser.add_argument(
        "--fairseq_root",
        type=str,
        help="absolute path to the fairseq",
    )
    parser.add_argument(
        "--shas_root",
        type=str,
        help="absolute path to the SHAS",
    )
    parser.add_argument(
        "--mwersegmenter_root",
        type=str,
        help="absolute path to the mwerSegmenter",
    )
    parser.add_argument(
        "--path_to_st_checkpoint",
        "-st_ckpt",
        type=str,
        required=True,
        help="absolute path to the st model checkpoint",
    )
    parser.add_argument(
        "--path_to_src_yaml",
        type=str,
        help="absolute path to the yaml of the corpus segmentation",
    )
    parser.add_argument(
        "--path_to_src_txt",
        type=str,
        help="absolute path to the source text of the corpus segmentation",
    )
    parser.add_argument(
        "--path_to_ref_txt",
        type=str,
        help="absolute path to the reference text of the corpus segmentation",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        choices=["BLEU"],
        default="BLEU",
        help="metrics to select better segments in the third stage",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--evaluate_data",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
