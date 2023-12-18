import gc
import os
import subprocess
import sys
import warnings
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from shlex import split

import hydra
import logzero
import numpy as np
import torch
import wandb
import yaml
from hydra.utils import instantiate
from lib.constants import ID_PAD, INPUT_SAMPLE_RATE, TARGET_SAMPLE_RATE
from lib.dataset import FixedDataloaderGenerator, FixedSegmentationDatasetNoTarget
from lib.datautils import AutoRegCollateFn, CollateFn
from lib.eval_scripts.format_generation_output import format_generation_output
from lib.eval_scripts.original_segmentation_to_xml import original_segmentation_to_xml
from lib.eval_scripts.prepare_custom_dataset import prepare_custom_dataset
from lib.eval_scripts.score import score_bertscore, score_bleurt, score_sacrebleu
from lib.evaluate import evaluate, infer
from lib.segment import pdac, pdac_with_logits, pthr, strm, update_yaml_content
from logzero import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm


def eval_st(config, infer_config, model, checkpoint_name, vocab, device):
    algo_conf = OmegaConf.to_object(infer_config.algorithm)
    algorithm = algo_conf.pop("tag")

    results = {}

    results_path = Path(config.results_path) / "eval_st" / checkpoint_name / algorithm
    results_path.mkdir(parents=True)

    cust_seg_yaml = results_path / infer_config.cust_seg_yaml

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

    yaml_content = []
    for wav_path in tqdm(
        sorted(list(Path(infer_config.infer_data.wav_dir).glob("*.wav")))
    ):
        # initialize a dataset for the fixed segmentation
        dataset = FixedSegmentationDatasetNoTarget(
            wav_path,
            infer_config.inference_segment_length,
            infer_config.inference_times,
        )
        sgm_frame_probs = None

        for inference_iteration in range(infer_config.inference_times):
            # create a dataloader for this fixed-length segmentation of the wav file
            dataset.fixed_length_segmentation(inference_iteration)
            dataloader = DataLoader(
                dataset,
                batch_size=infer_config.batch_size,
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

        sgm_frame_probs /= infer_config.inference_times

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

    # segment audio
    with open(cust_seg_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=True)
    results[f"eval_st_n_segments_{algorithm}"] = len(yaml_content)

    # prepare the tsv file from the custom segmentation yaml
    prepare_custom_dataset(
        cust_seg_yaml,
        infer_config.infer_data.wav_dir,
        infer_config.infer_data.tgt_lang,
        0,
    )

    # translate using the Speech Trasnlation model
    fairseq_generate_cmd = (
        f"fairseq-generate {results_path}"
        " --task speech_text_joint_to_text"
        " --max-tokens 100000"
        " --max-source-positions 12000"
        " --nbest 1"
        " --batch-size 128"
        f" --path {infer_config.st_model_dir}/{infer_config.st_ckpt}"
        f" --gen-subset {Path(infer_config.cust_seg_yaml).stem}"
        f" --config-yaml {infer_config.st_model_dir}/config.yaml"
        " --beam 5"
        " --lenpen 1.0"
        " --skip-invalid-size-inputs-valid-test"
        f" --user-dir {infer_config.fairseq_root}/examples/speech_text_joint_to_text"
        f" --load-speech-only > {results_path}/translations.txt"
    )
    subprocess.run(fairseq_generate_cmd, shell=True)

    # extract raw hypotheses from fairseq-generate output
    format_generation_output(f"{results_path}/translations.txt")

    original_segmentation_to_xml(
        infer_config.infer_data.orig_seg_yaml,
        infer_config.infer_data.orig_src_txt,
        infer_config.infer_data.orig_tgt_txt,
        results_path,
    )

    # align the hypotheses with the references
    split_name = Path(infer_config.infer_data.orig_seg_yaml).stem
    sysid = Path(infer_config.st_model_dir).stem
    mwersegment_cmd = (
        f"{infer_config.mwersegmenter_root}/segmentBasedOnMWER.sh"
        f" {results_path}/{split_name}.{infer_config.infer_data.src_lang}.xml"
        f" {results_path}/{split_name}.{infer_config.infer_data.tgt_lang}.xml"
        f" {results_path}/translations_formatted.txt"
        f" {sysid} {infer_config.infer_data.tgt_lang}"
        f" {results_path}/translations_aligned.xml normalize 1"
    )
    subprocess.run(mwersegment_cmd, shell=True)

    # obtain scores of the aligned hypothesis and references
    if "bleu" in infer_config.st_metrics:
        from sacrebleu.metrics.bleu import BLEUScore

        bleu = score_sacrebleu(
            f"{results_path}/__mreference",
            f"{results_path}/__segments",
        )
        bleu_print = str(bleu)
        bleu = bleu.score
        with open(f"{results_path}/score.sacrebleu", "w") as f:
            f.write(bleu_print)
        results[f"eval_st_bleu_{algorithm}"] = bleu

    if "bertscore" in infer_config.st_metrics:
        bertscore = score_bertscore(
            f"{results_path}/__mreference",
            f"{results_path}/__segments",
            f"{infer_config.infer_data.tgt_lang}",
        )
        p = bertscore[0]
        r = bertscore[1]
        f1 = bertscore[2]
        bertscore_print = "BERTScore (P/R/F1) = " f"{p:.4f}/{r:.4f}/{f1:.4f}"
        with open(f"{results_path}/score.bertscore", "w") as f:
            f.write(bertscore_print)
        results[f"eval_st_bertscore_p_{algorithm}"] = p
        results[f"eval_st_bertscore_r_{algorithm}"] = r
        results[f"eval_st_bertscore_f1_{algorithm}"] = f1

    if "bleurt" in infer_config.st_metrics:
        bleurt = score_bleurt(
            f"{results_path}/__mreference",
            f"{results_path}/__segments",
            f"{infer_config.bleurt_path}",
        )
        bleurt_print = f"BLEURT (Average) = {bleurt:.4f}"
        with open(f"{results_path}/score.bleurt", "w") as f:
            f.write(bleurt_print)
        results[f"eval_st_bleurt_{algorithm}"] = bleurt

    return results, model


def train(config: DictConfig) -> None:
    results_path = Path(os.getcwd()) / config.exp_name
    checkpoints_path = results_path / "ckpts"
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    # add a new key to OmegaConf obj for logging
    OmegaConf.set_struct(config, False)
    config["results_path"] = str(results_path)

    if config.log_wandb:
        run = wandb.init(
            project=config.project_name,
            config=config,
            name=config.exp_name,
            notes=config.notes,
            group=config.group,
            dir=str(results_path),
        )

    # number of cpu and gpu devices
    n_gpu = torch.cuda.device_count()
    n_cpu = cpu_count()
    num_workers = min(4, n_cpu // 2)
    device_list = [torch.device(f"cuda:{i}") for i in range(n_gpu)]
    main_device = device_list[0] if n_gpu > 0 else torch.device("cpu")
    print(f"Number of cuda devices: {n_gpu} | Number of CPU cores: {n_cpu}")
    print(f"Main device: {main_device}")
    print(f"Parallel devices = {device_list}")

    # adjust batch size for number of gpus
    effective_batch_size = config.batch_size * n_gpu if n_gpu else config.batch_size

    autoregression = config.task.autoregression

    device_conf = DictConfig(
        {
            "batch_size": effective_batch_size,
            "num_workers": num_workers,
        }
    )

    # load vocabulary
    if config.task.vocab:
        vocab = instantiate(config.task.vocab)
        config.task.model["vocab_size"] = vocab.vocab_size
    else:
        vocab = None

    # train dataloader generator
    train_dataloader_generator = instantiate(
        OmegaConf.merge(
            config.task.train_generator,
            config.data.train,
            device_conf,
        ),
        autoregression=autoregression,
        vocab=vocab,
    )

    # eval dataloader generator
    eval_dataloader_generator = instantiate(
        OmegaConf.merge(
            config.task.eval_generator,
            config.data.eval,
            device_conf,
        ),
        autoregression=autoregression,
        vocab=vocab,
    )

    model = instantiate(
        config.task.model,
    ).to(main_device)

    # finetune
    if config.finetune_from_model:
        if config.task.model.finetune_wav2vec:
            model.load_state_dict(torch.load(config.finetune_from_model)["state_dict"])
        else:
            model.seg_model.load_state_dict(
                torch.load(config.finetune_from_model)["state_dict"]
            )

    # view model summaries
    logger.info(
        summary(
            model,
            input_size=[
                (config.batch_size, int(INPUT_SAMPLE_RATE * config.segment_length)),
                (config.batch_size, int(INPUT_SAMPLE_RATE * config.segment_length)),
                (config.batch_size, int(TARGET_SAMPLE_RATE * config.segment_length)),
            ],
            depth=9,
        )
    )

    # optionally parallel models
    if len(device_list) > 1:
        model = torch.nn.DataParallel(
            model, device_ids=device_list, output_device=main_device
        )

    if config.log_wandb:
        wandb.watch(model, log="all", log_freq=config.print_every_steps)

    # get first train dataloader to approximate total steps during training
    if isinstance(train_dataloader_generator, FixedDataloaderGenerator):
        # training with FixedDataloaderGenerator
        train_dataloader = train_dataloader_generator.generate("", 0)
    else:
        train_dataloader = train_dataloader_generator.generate()
    total_steps_approx = int(
        config.max_epochs * len(train_dataloader) / config.update_freq * 1.01
    )

    # Adam with cosine annealing
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, total_steps_approx)

    if config.save_ckpts:
        ckpt_list = []
        if config.keep_best_ckpt:
            best_metric = config.best_ckpt_metric
            best_score = 0
            best_checkpoint = ""

    global_step = 0
    for epoch in range(config.max_epochs):
        print(f"Starting epoch {epoch} ...")

        if epoch:
            if isinstance(train_dataloader_generator, FixedDataloaderGenerator):
                # training with FixedDataloaderGenerator
                train_dataloader = train_dataloader_generator.generate("", 0)
            else:
                train_dataloader = train_dataloader_generator.generate()

        # create loss_fn
        loss_conf = OmegaConf.to_object(config.task.loss)
        loss_tag = loss_conf.pop("tag")
        ma_window = loss_conf.pop("ma_window")
        if loss_tag == "bce":
            logger.info(
                f"pos_class_percentage = {train_dataloader.dataset.pos_class_percentage}"
            )
            if not config.task.loss.pos_weight:
                pos_weight = 1 - train_dataloader.dataset.pos_class_percentage
                loss_conf["pos_weight"] = torch.tensor(pos_weight)
                logger.info(f"pos_weight = {pos_weight} (1 - pos_class_percentage)")
            else:
                loss_conf["pos_weight"] = torch.tensor(config.task.loss.pos_weight)
                logger.info(
                    f"pos_weight = {config.task.loss.pos_weight} (config.task.loss.pos_weight)"
                )
        elif loss_tag in ["ce", "ssl"]:
            # [TODO] weight
            loss_conf["ignore_index"] = vocab.pad_token_id
        else:
            raise NotImplementedError()
        loss_fn = instantiate(loss_conf)

        model.train()

        steps_in_epoch = len(train_dataloader)
        all_losses, all_preds, all_targets = [], [], []

        for step, batch in enumerate(iter(train_dataloader), start=1):
            global_step += 1

            model_args = {}
            model_args["audio"] = batch["audio"].to(main_device)
            model_args["in_mask"] = batch["in_mask"].to(main_device)

            if not autoregression:
                out_mask = batch["out_mask"].to(main_device)
                model_args["out_mask"] = out_mask
                target = batch["target"].to(main_device)
            else:
                model_args["src_pad_mask"] = batch["src_pad_mask"].to(main_device)
                model_args["tgt_pad_mask"] = batch["tgt_pad_mask"].to(main_device)
                model_args["tgt_mask"] = batch["tgt_mask"].to(main_device)
                model_args["target"] = batch["in_target"].to(main_device)
                target = batch["out_target"].to(main_device)

            model_output = model(**model_args)

            if loss_tag in ["ssl"]:
                target_ctc = torch.argmax(model_output[0], dim=-1)
                logits = model_output[1]
            else:
                logits = model_output

            # calculate loss
            if loss_tag == "bce":
                # some times the output of wav2vec is 1 frame larger/smaller
                # correct for these cases
                size1 = logits.shape[1]
                size2 = target.shape[1]
                if size1 != size2:
                    if size1 < size2:
                        logger.info(
                            (
                                f"logits length ({logits.shape[1]}) is "
                                f"smaller than target's ({target.shape[1]})."
                            )
                        )
                        target = target[:, :-1]
                        out_mask = out_mask[:, :-1]
                    else:
                        logger.info(
                            (
                                f"logits length ({logits.shape[1]}) is "
                                f"larger than target's ({target.shape[1]})."
                            )
                        )
                        logits = logits[:, :-1]
                loss_per_point = loss_fn(logits, target)
                loss_per_point[~out_mask] = 0
                if ma_window:
                    from lib.constants import WAV2VEC_FRAME_LEN
                    from lib.segment import moving_average

                    frame_length = WAV2VEC_FRAME_LEN / 1000
                    ma_window_steps = int(ma_window / frame_length)
                    target_ma = []
                    for i in range(target.size(0)):
                        target_ma.append(
                            moving_average(
                                target[i].cpu().detach().numpy().copy(),
                                ma_window_steps,
                            )
                        )
                    target_ma = np.array(target_ma)
                    target_ma = torch.from_numpy(target_ma.astype(np.float32)).to(
                        main_device
                    )
                    ma_weight = 1 - torch.abs(target - target_ma)
                    loss_per_point = torch.mul(loss_per_point, ma_weight)

                loss = loss_per_point.sum(dim=1).mean()
            elif loss_tag == "ce":
                loss_per_point = loss_fn(
                    logits.view(-1, vocab.vocab_size), target.view(-1).to(int)
                )
                loss = loss_per_point.sum(dim=0).mean()
            elif loss_tag == "ssl":
                target_ctc += vocab.n_special_tokens
                nb_mask = target != vocab.nonboundary_token_id
                masked_target = nb_mask * target
                masked_target_ctc = ~nb_mask * target_ctc
                target_ssl = masked_target + masked_target_ctc
                loss_per_point = loss_fn(
                    logits.view(-1, vocab.vocab_size), target_ssl.view(-1).to(int)
                )
                loss = loss_per_point.sum(dim=0).mean()
            else:
                raise NotImplementedError()

            # accumulate loss
            (loss / config.update_freq).backward()

            # apply optimizer
            if (not step % config.update_freq) or (step == steps_in_epoch):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # store for summary
            all_losses.append(loss.detach().cpu().numpy().item())
            if loss_tag == "bce":
                all_preds.extend(
                    (torch.sigmoid(logits) >= 0.5)[out_mask]
                    .view(-1)
                    .detach()
                    .cpu()
                    .tolist()
                )
                all_targets.extend(
                    (target >= 0.5)[out_mask].view(-1).detach().cpu().tolist()
                )
            elif loss_tag in ["ce", "ssl"]:
                spe_token_mask = (target == vocab.boundary_token_id) + (
                    target == vocab.nonboundary_token_id
                )
                pred = torch.argmax(logits, dim=2)
                pred = (pred != vocab.boundary_token_id) * torch.ones(pred.shape).to(
                    main_device
                )
                all_preds.extend(pred[spe_token_mask].detach().cpu().tolist())
                all_targets.extend(target[spe_token_mask].detach().cpu().tolist())
            else:
                raise NotImplementedError()

            if (not step % config.print_every_steps) or (step == steps_in_epoch):
                accuracy = f1_score(all_targets, all_preds, average="micro")
                f1 = f1_score(all_targets, all_preds, average="binary")
                precision = precision_score(all_targets, all_preds)
                recall = recall_score(all_targets, all_preds)
                avg_loss = np.mean(all_losses)
                print(
                    "[{}]: Step {}/{}, loss = {:.4f}, accuracy {:.4f}, f1 {:.4f}, "
                    "precision {:.4f}, recall {:.4f}, lr {:.6f}".format(
                        datetime.now().time().replace(microsecond=0),
                        str(step).zfill(len(str(steps_in_epoch))),
                        steps_in_epoch,
                        avg_loss,
                        accuracy,
                        f1,
                        precision,
                        recall,
                        scheduler.get_last_lr()[0],
                    )
                )

                if config.log_wandb:
                    wandb_dict = {
                        "epoch": epoch,
                        "loss": avg_loss,
                        "accuracy": accuracy,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                        "lr": scheduler.get_last_lr()[0],
                    }
                    wandb.log(wandb_dict, step=global_step)

                all_losses, all_targets, all_preds = [], [], []

            if not global_step % config.save_every_steps:
                print(
                    "[{}]: Epoch {} - Step {}: Starting evaliation ...".format(
                        datetime.now().time().replace(microsecond=0),
                        epoch,
                        global_step,
                    )
                )
                model.eval()
                results = evaluate(
                    eval_dataloader_generator,
                    model,
                    main_device,
                    autoregression=autoregression,
                    loss_tag=loss_tag,
                    vocab=vocab,
                    loss_fn=loss_fn,
                )
                model.train()
                if config.log_wandb:
                    wandb.log(results)
                print(results)

                if config.perform_st_evaluation:
                    if config.st_eval is not None:
                        st_results, model = eval_st(
                            config,
                            config.st_eval,
                            model,
                            f"epoch-{epoch}_step-{global_step}",
                            vocab,
                            main_device,
                        )
                    if config.st_eval_online is not None:
                        st_results_online, model = eval_st(
                            config,
                            config.st_eval_online,
                            model,
                            f"epoch-{epoch}_step-{global_step}",
                            vocab,
                            main_device,
                        )
                    model.train()
                    if config.log_wandb:
                        if config.st_eval is not None:
                            wandb.log(st_results)
                        if config.st_eval_online is not None:
                            wandb.log(st_results_online)

                if config.save_ckpts:
                    new_checkpoint = (
                        checkpoints_path / f"epoch-{epoch}_step-{global_step}.pt"
                    )
                    if config.task.model.finetune_wav2vec:
                        torch.save(
                            {
                                "state_dict": model.module.state_dict()
                                if n_gpu > 1
                                else model.state_dict(),
                            },
                            new_checkpoint,
                        )
                    else:  # do not store wav2vec2 state_dict
                        torch.save(
                            {
                                "state_dict": model.module.seg_model.state_dict()
                                if n_gpu > 1
                                else model.seg_model.state_dict(),
                            },
                            new_checkpoint,
                        )
                    ckpt_list.append(new_checkpoint)
                    if len(ckpt_list) > config.keep_last_ckpts:
                        os.remove(ckpt_list.pop(0))

                    # save the best checkpoint
                    if config.keep_best_ckpt:
                        curr_score = results[best_metric]
                        if curr_score > best_score:
                            if os.path.isfile(best_checkpoint):
                                os.remove(best_checkpoint)
                            best_checkpoint = str(new_checkpoint).replace(
                                ".pt", f"_best_{best_metric}.pt"
                            )
                            best_score = curr_score
                            if config.task.model.finetune_wav2vec:
                                torch.save(
                                    {
                                        "state_dict": model.module.state_dict()
                                        if n_gpu > 1
                                        else model.state_dict(),
                                    },
                                    best_checkpoint,
                                )
                            else:
                                torch.save(
                                    {
                                        "state_dict": model.module.seg_model.state_dict()
                                        if n_gpu > 1
                                        else model.seg_model.state_dict(),
                                    },
                                    best_checkpoint,
                                )

        print(
            "[{}]: Epoch {}: Starting evaliation ...".format(
                datetime.now().time().replace(microsecond=0),
                epoch,
            )
        )
        model.eval()
        results = evaluate(
            eval_dataloader_generator,
            model,
            main_device,
            autoregression=autoregression,
            loss_tag=loss_tag,
            vocab=vocab,
            loss_fn=loss_fn,
        )
        if config.log_wandb:
            wandb.log(results)
        print(results)

        if config.perform_st_evaluation:
            if config.st_eval is not None:
                st_results, model = eval_st(
                    config,
                    config.st_eval,
                    model,
                    f"epoch-{epoch}",
                    vocab,
                    main_device,
                )
            if config.st_eval_online is not None:
                st_results_online, model = eval_st(
                    config,
                    config.st_eval_online,
                    model,
                    f"epoch-{epoch}",
                    vocab,
                    main_device,
                )
            model.train()
            if config.log_wandb:
                if config.st_eval is not None:
                    wandb.log(st_results)
                if config.st_eval_online is not None:
                    wandb.log(st_results_online)

        if config.save_ckpts:
            new_checkpoint = checkpoints_path / f"epoch-{epoch}.pt"
            if config.task.model.finetune_wav2vec:
                torch.save(
                    {
                        "state_dict": model.module.state_dict()
                        if n_gpu > 1
                        else model.state_dict(),
                    },
                    new_checkpoint,
                )
            else:  # do not store wav2vec2 state_dict
                torch.save(
                    {
                        "state_dict": model.module.seg_model.state_dict()
                        if n_gpu > 1
                        else model.seg_model.state_dict(),
                    },
                    new_checkpoint,
                )
            ckpt_list.append(new_checkpoint)
            if len(ckpt_list) > config.keep_last_ckpts:
                os.remove(ckpt_list.pop(0))

            # save the best checkpoint
            if config.keep_best_ckpt:
                curr_score = results[best_metric]
                if curr_score > best_score:
                    if os.path.isfile(best_checkpoint):
                        os.remove(best_checkpoint)
                    best_checkpoint = str(new_checkpoint).replace(
                        ".pt", f"_best_{best_metric}.pt"
                    )
                    best_score = curr_score
                    if config.task.model.finetune_wav2vec:
                        torch.save(
                            {
                                "state_dict": model.module.state_dict()
                                if n_gpu > 1
                                else model.state_dict(),
                            },
                            best_checkpoint,
                        )
                    else:
                        torch.save(
                            {
                                "state_dict": model.module.seg_model.state_dict()
                                if n_gpu > 1
                                else model.seg_model.state_dict(),
                            },
                            best_checkpoint,
                        )

    if config.log_wandb:
        run.finish()


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


@hydra.main(config_path="conf", config_name="train")
def main(config: DictConfig) -> None:
    init(config)
    train(config)


if __name__ == "__main__":
    main()
