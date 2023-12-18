from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm


def infer(
    model,
    dataloader,
    main_device,
    autoregression,
    loss_tag,
    vocab=None,
    loss_fn=None,
) -> Tuple[np.array, np.array]:
    """Does inference for a single wav file"""

    duration_outframes = dataloader.dataset.duration_outframes

    talk_probs = np.empty(duration_outframes)
    talk_probs[:] = np.nan
    talk_targets = np.zeros(duration_outframes)
    if vocab:
        talk_logits = np.empty((duration_outframes, vocab.vocab_size))
    else:
        talk_logits = np.empty(duration_outframes)
    talk_logits[:] = np.nan

    loss = None
    avg_loss = None
    all_losses = []

    for batch in iter(dataloader):
        audio = batch["audio"].to(main_device)
        in_mask = batch["in_mask"].to(main_device)

        included = batch["included"]
        starts = batch["starts"]
        ends = batch["ends"]

        if not autoregression:
            out_mask = batch["out_mask"].to(main_device)
            if batch["target"] is not None:
                targets = batch["target"].to(main_device)
            else:
                targets = None
        else:
            raise NotImplementedError()
            src_pad_mask = batch["src_pad_mask"].to(main_device)
            tgt_pad_mask = batch["tgt_pad_mask"].to(main_device)
            tgt_mask = batch["tgt_mask"].to(main_device)
            in_target = batch["in_target"].to(main_device)
            targets = batch["out_target"]
            out_target = batch["out_target"].to(main_device)

        with torch.no_grad():
            _, wav2vec_hidden = model.wav2vec_model(audio, in_mask)

            # some times the output of wav2vec is 1 frame larger/smaller
            # correct for these cases
            size1 = wav2vec_hidden.shape[1]
            size2 = out_mask.shape[1]
            if size1 != size2:
                if size1 < size2:
                    out_mask = out_mask[:, :-1]
                    ends = [e - 1 for e in ends]
                else:
                    wav2vec_hidden = wav2vec_hidden[:, :-1, :]

            logits = model.seg_model(wav2vec_hidden, out_mask)

            if loss_tag == "bce":
                if loss_fn:
                    size = min(logits.shape[1], targets.shape[1])
                    logits = logits[:, :size]
                    targets = targets[:, :size]
                    loss_per_point = loss_fn(logits, targets)
                    loss_per_point[~out_mask] = 0
                    loss = loss_per_point.sum(dim=1).mean()
                probs = torch.sigmoid(
                    logits
                )  # [TODO] return 1 - sigmoid (boundary prob)
            elif loss_tag in ["ce", "ssl"]:
                # RM probs = 1 - torch.nn.functional.softmax(logits, dim=-1)[:, :, 0]
                probs = torch.nn.functional.softmax(logits, dim=-1)[:, :, 0]
            else:
                raise NotImplementedError()
            probs[~out_mask] = 0
            logits[~out_mask] = 0

        if loss:
            all_losses.append(loss.detach().cpu().numpy().item())

        probs = probs.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()

        # fill-in the probabilities and targets for the talk
        for i in range(len(probs)):
            start, end = starts[i], ends[i]
            if included[i] and end > start:
                duration = end - start
                talk_probs[start:end] = probs[i, :duration]
                talk_logits[start:end] = logits[i, :duration]
                if targets is not None:
                    targets = targets.detach().cpu()
                    talk_targets[start:end] = targets[i, :duration].numpy()
            elif not included[i]:
                talk_probs[start:end] = 0  # [TODO] 0 if p=speech, 1 if p=boundary
                talk_logits[start:end] = 0

    if loss:
        avg_loss = np.mean(all_losses)

    # account for the rare incident that a frame didnt have a prediction
    # fill-in those frames with the average of the surrounding frames
    nan_idx = np.where(np.isnan(talk_probs))[0]
    for j in nan_idx:
        talk_probs[j] = np.nanmean(
            talk_probs[max(0, j - 2) : min(duration_outframes, j + 3)]
        )
        talk_logits[j] = np.nanmean(
            talk_logits[max(0, j - 2) : min(duration_outframes, j + 3)]
        )

    return talk_probs, talk_logits, talk_targets, avg_loss


def evaluate(
    dataloader_generator,
    model,
    main_device,
    autoregression,
    loss_tag,
    vocab,
    loss_fn=None,
) -> dict[str, float]:
    """Does inference and evaluation for a dev/test set"""

    all_preds, all_targets = np.array([]), np.array([])
    talk_ids = dataloader_generator.get_talk_ids()

    for talk_id in tqdm(talk_ids):
        inference_times = dataloader_generator.dataset.inference_times
        probs, targets = None, None

        all_losses = []

        # multiple inferences on difference fixed-length segmentations
        for iteration in range(inference_times):
            # get dataloader object for this specific segmentation of the wav
            dataloader = dataloader_generator.generate(talk_id, iteration)

            # get probabilities and targets
            p, l, t, loss = infer(
                model,
                dataloader,
                main_device,
                autoregression,
                loss_tag,
                vocab,
                loss_fn,
            )
            if probs is None:
                probs = p.copy()
                logits = l.copy()
                targets = t.copy()
                if loss:
                    losses = loss.copy()
                else:
                    losses = None
            else:
                logits += l
                probs += p
                if loss:
                    losses += loss

        probs /= inference_times
        if losses:
            losses /= inference_times

        # predictions for the wav
        if loss_tag == "bce":
            preds = probs / inference_times > 0.5

        elif loss_tag in ["ce", "ssl"]:
            preds = np.argmax(logits, axis=-1) == vocab.boundary_token_id
            spe_token_mask = targets != vocab.pad_token_id  # [TODO] may be unnecessary
            targets = targets * spe_token_mask
            eval_loss = None  # [TODO] implementation
        else:
            raise NotImplementedError()

        all_preds = np.append(all_preds, preds)
        all_targets = np.append(all_targets, targets)
        if loss_fn:
            all_losses.append(losses)

    all_targets = all_targets.astype(bool)
    all_preds = all_preds.astype(bool)
    if loss_fn:
        eval_loss = np.mean(all_losses)

    results = {
        f"eval_accuracy": round(f1_score(all_targets, all_preds, average="micro"), 4),
        f"eval_f1": round(f1_score(all_targets, all_preds, average="binary"), 4),
        f"eval_precision": round(precision_score(all_targets, all_preds), 4),
        f"eval_recall": round(recall_score(all_targets, all_preds), 4),
    }
    if eval_loss:
        results["eval_loss"] = eval_loss

    return results
