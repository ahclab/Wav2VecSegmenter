import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)

from constants import HIDDEN_SIZE, TARGET_SAMPLE_RATE, WAV2VEC_FRAME_LEN


@dataclass
class Segment:
    start: float
    end: float
    probs: np.array = None
    logits: np.array = None
    decimal: int = 6

    @property
    def duration(self):
        return float(round((self.end - self.start) / TARGET_SAMPLE_RATE, self.decimal))

    @property
    def offset(self):
        return float(round(self.start / TARGET_SAMPLE_RATE, self.decimal))

    @property
    def offset_plus_duration(self):
        return round(self.offset + self.duration, self.decimal)


def trim(sgm: Segment, threshold: float) -> Segment:
    """reduces the segment to between the first and last points that are above the threshold
    Args:
        sgm (Segment): a segment
        threshold (float): probability threshold
    Returns:
        Segment: new reduced segment
    """
    included_indices = np.where(sgm.probs >= threshold)[0]

    # return empty segment
    if not len(included_indices):
        return Segment(sgm.start, sgm.start, probs=np.empty([0]))

    i = included_indices[0]
    j = included_indices[-1] + 1

    sgm = Segment(sgm.start + i, sgm.start + j, probs=sgm.probs[i:j])

    return sgm


def argtrim(sgm: Segment, vocab) -> Segment:
    """reduces the segment to between the first and last points that are not the boundary
    Args:
        sgm (Segment): a segment
        vocab (BaseVocabulary): vocabulary
    Returns:
        Segment: new reduced segment
    """
    preds = np.argmax(sgm.logits, axis=-1)
    included_indices = np.where(preds != vocab.boundary_token_id)[0]

    # return empty segment
    if not len(included_indices):
        return Segment(sgm.start, sgm.start, probs=np.empty([0]), logits=np.empty([0]))

    i = included_indices[0]
    j = included_indices[-1] + 1

    sgm = Segment(
        sgm.start + i, sgm.start + j, probs=sgm.probs[i:j], logits=sgm.logits[i:j]
    )

    return sgm


def soft_trim(
    sgm: Segment, boundary_threshold: float, trim_threshold: float
) -> Segment:
    """trimming for segment tree generation
    - reduces the segment to between the first and last points that are above the trim_threshold
    - outside the segment between the first and last points that are above the threshold cannot be a segment boundary
    Args:
        sgm (Segment): a segment
        boundary_threshold (float): probability threshold
        trim_threshold (float): probability threshold to trim,
    Returns:
        Segment: new reduced segment
    """
    boundary_cand_indices = np.where(sgm.probs >= boundary_threshold)[0]
    included_indices = np.where(sgm.probs >= trim_threshold)[0]

    # return empty segment
    if not len(boundary_cand_indices):
        return Segment(sgm.start, sgm.start, probs=np.empty([0]))

    # filter by boundary threshold
    sgm.probs[: boundary_cand_indices[0]] = 1
    sgm.probs[boundary_cand_indices[-1] + 1 :] = 1

    i = included_indices[0]
    j = included_indices[-1] + 1

    sgm = Segment(sgm.start + i, sgm.start + j, probs=sgm.probs[i:j])

    return sgm


def split_and_trim(
    sgm: Segment, split_idx: int, threshold: float
) -> tuple[Segment, Segment]:
    """splits the input segment at the split_idx and then trims and returns the two resulting segments
    Args:
        sgm (Segment): input segment
        split_idx (int): index to split the input segment
        threshold (float): probability threshold
    Returns:
        tuple[Segment, Segment]: the two resulting segments
    """

    probs_a = sgm.probs[:split_idx]
    sgm_a = Segment(sgm.start, sgm.start + len(probs_a), probs_a)

    probs_b = sgm.probs[split_idx + 1 :]
    sgm_b = Segment(sgm_a.end + 1, sgm.end, probs_b)

    sgm_a = trim(sgm_a, threshold)
    sgm_b = trim(sgm_b, threshold)

    return sgm_a, sgm_b


def split_and_argtrim(sgm: Segment, split_idx: int, vocab) -> tuple[Segment, Segment]:
    """splits the input segment at the split_idx and then trims and returns the two resulting segments
    Args:
        sgm (Segment): input segment
        split_idx (int): index to split the input segment
        vocab (BaseVocabulary)
    Returns:
        tuple[Segment, Segment]: the two resulting segments
    """

    probs_a = sgm.probs[:split_idx]
    logits_a = sgm.logits[:split_idx]
    sgm_a = Segment(sgm.start, sgm.start + len(probs_a), probs_a, logits_a)

    probs_b = sgm.probs[split_idx + 1 :]
    logits_b = sgm.logits[split_idx + 1 :]
    sgm_b = Segment(sgm_a.end + 1, sgm.end, probs_b, logits_b)

    sgm_a = argtrim(sgm_a, vocab)
    sgm_b = argtrim(sgm_b, vocab)

    return sgm_a, sgm_b


def split_and_softtrim(
    sgm: Segment, split_idx: int, boundary_threshold: float, trim_threshold: float
) -> tuple[Segment, Segment]:
    """splits the input segment at the split_idx and then soft_trims and returns the two resulting segments
    Args:
        sgm (Segment): input segment
        split_idx (int): index to split the input segment
        boundary_threshold (float): probability threshold
        trim_threshold (float): probability threshold to trim
    Returns:
        tuple[Segment, Segment]: the two resulting segments
    """

    probs_a = sgm.probs[:split_idx]
    sgm_a = Segment(sgm.start, sgm.start + len(probs_a), probs_a)

    probs_b = sgm.probs[split_idx + 1 :]
    sgm_b = Segment(sgm_a.end + 1, sgm.end, probs_b)

    sgm_a = soft_trim(sgm_a, boundary_threshold, trim_threshold)
    sgm_b = soft_trim(sgm_b, boundary_threshold, trim_threshold)

    return sgm_a, sgm_b


def pdac(
    probs: np.array,
    max_segment_length: float = 18,
    min_segment_length: float = 0.2,
    threshold: float = 0.5,
) -> list[Segment]:
    """applies the probabilistic Divide-and-Conquer algorithm to split an audio
    into segments satisfying the max-segment-length and min-segment-length conditions
    Args:
        probs (np.array): the binary frame-level probabilities
            output by the segmentation-frame-classifier
        max_segment_length (float): the maximum length of a segment
        min_segment_length (float): the minimum length of a segment
        threshold (float): probability threshold
    Returns:
        list[Segment]: resulting segmentation
    """

    segments = []
    sgm = Segment(0, len(probs), probs=probs)
    sgm = trim(sgm, threshold)

    def recusrive_split(sgm):
        if sgm.duration < max_segment_length:
            segments.append(sgm)
        else:
            j = 0
            sorted_indices = np.argsort(sgm.probs)
            while j < len(sorted_indices):
                split_idx = sorted_indices[j]
                split_prob = sgm.probs[split_idx]
                if split_prob > threshold:
                    segments.append(sgm)
                    break

                sgm_a, sgm_b = split_and_trim(sgm, split_idx, threshold)
                if (
                    sgm_a.duration > min_segment_length
                    and sgm_b.duration > min_segment_length
                ):
                    recusrive_split(sgm_a)
                    recusrive_split(sgm_b)
                    break
                j += 1
            else:
                segments.append(sgm)

    recusrive_split(sgm)

    return segments


def pdac_with_logits(
    probs: np.array,
    logits: np.array,
    vocab,
    max_segment_length: float = 18,
    min_segment_length: float = 0.2,
) -> list[Segment]:
    """applies the probabilistic Divide-and-Conquer algorithm to split an audio
    into segments satisfying the max-segment-length and min-segment-length conditions
    Args:
        probs (np.array): the binary frame-level probabilities
            output by the segmentation-frame-classifier
        logits (np.array): logits
        vocab (BaseVocabulary)
        max_segment_length (float): the maximum length of a segment
        min_segment_length (float): the minimum length of a segment
    Returns:
        list[Segment]: resulting segmentation
    """

    segments = []
    sgm = Segment(0, len(logits), probs=probs, logits=logits)
    sgm = argtrim(sgm, vocab)

    def recusrive_split(sgm):
        if sgm.duration < max_segment_length:
            segments.append(sgm)
        else:
            j = 0
            sorted_indices = np.argsort(sgm.probs)[::-1]
            while j < len(sorted_indices):
                split_idx = sorted_indices[j]
                split_prob = sgm.probs[split_idx]

                sgm_a, sgm_b = split_and_argtrim(sgm, split_idx, vocab)
                if (
                    sgm_a.duration > min_segment_length
                    and sgm_b.duration > min_segment_length
                ):
                    recusrive_split(sgm_a)
                    recusrive_split(sgm_b)
                    break
                j += 1
            else:
                segments.append(sgm)

    recusrive_split(sgm)

    return segments


def visualize_tree(tree: list[Segment], depth: int = 999):
    layer = 0
    nextp = 2 ** (layer + 1) - 1
    print(f"layer({layer:03}): ", end="")
    for i, seg in enumerate(tree):
        if i >= nextp:
            print("\n")
            layer += 1
            nextp = 2 ** (layer + 1) - 1
            if layer > depth:
                break
            print(f"layer({layer:03}): ", end="")
        print(f"[{seg.offset}+{seg.duration}] ", end="")
    print("\n")


def pdac_tree(
    probs: np.array,
    max_segment_length: float = 18,
    min_segment_length: float = 0.2,
    boundary_threshold: float = 0.5,
    trim_threshold: float = 0,
    tree_depth: int = 20,
) -> list[Segment]:
    """generate binary tree by probabilistic Divide-and-Conquer
    Args:
        probs (np.array): the binary frame-level probabilities
            output by the segmentation-frame-classifier
        max_segment_length (float): the maximum length of a segment
        min_segment_length (float): the minimum length of a segment
        boundary_threshold (float): probability threshold
        trim_threshold (float): probability threshold to trim
        tree_depth (int): upper bounda of binary tree depth
    Returns:
        list[Segment]: resulting segmentation
    """

    segments = []
    sgm = Segment(0, len(probs), probs=probs)
    sgm = soft_trim(sgm, boundary_threshold, trim_threshold)
    # DEBUG    sgm = trim(sgm, boundary_threshold)
    tree = [sgm]
    cond = [True]

    if len(sgm.probs) == 0:
        logging.warning("No segments found")
        return tree

    layer = 0  # current layer
    p = 2**layer - 1  # pointer of tree (0, 1, 3, 7, ...)
    while any(cond):
        if layer >= tree_depth:
            break
        for j, curr_sgm in enumerate(tree[p:]):
            logging.debug(f"layer={layer}, index of tree={p+j}")
            if cond[j]:
                split_idx = np.argsort(curr_sgm.probs)[0]
                if curr_sgm.probs[split_idx] == 1:
                    tree.append(
                        Segment(curr_sgm.start, curr_sgm.start, probs=np.empty([0]))
                    )
                    tree.append(
                        Segment(curr_sgm.start, curr_sgm.start, probs=np.empty([0]))
                    )
                else:
                    sgm_a, sgm_b = split_and_softtrim(
                        curr_sgm,
                        split_idx,
                        boundary_threshold,
                        trim_threshold,
                    )
                    if sgm_a.duration >= min_segment_length:
                        tree.append(sgm_a)
                    else:
                        tree.append(
                            Segment(sgm_a.start, sgm_a.start, probs=np.empty([0]))
                        )
                    if sgm_b.duration >= min_segment_length:
                        tree.append(sgm_b)
                    else:
                        tree.append(
                            Segment(sgm_b.start, sgm_b.start, probs=np.empty([0]))
                        )
            else:
                tree.append(
                    Segment(curr_sgm.start, curr_sgm.start, probs=np.empty([0]))
                )
                tree.append(
                    Segment(curr_sgm.start, curr_sgm.start, probs=np.empty([0]))
                )
        layer += 1
        p = 2**layer - 1
        cond = [sgm.duration >= min_segment_length for sgm in tree[p:]]
        logging.debug(f"cond.count(True)/len(cond)={cond.count(True)}/{len(cond)}")

    # DEBUG    visualize_tree(tree, depth=2)

    return tree


def get_segments(
    splitted_predictions: list[str],
    frame_length: float,
) -> list[dict]:
    """
    Args:
        splitted_predictions (list[str]): the splitted predictions from either STRM or DAC
        frame_length (float): the length of the predictive frame
    Returns:
        list[Segment]: segmentation
    """

    total_duration_frames = len("".join(splitted_predictions))

    segments = []
    offset = 0

    # expand each segment by a few seconds (0.06 s)
    minu_frame = TARGET_SAMPLE_RATE * 0.06

    for seg in splitted_predictions:
        if not is_pause(seg):
            start = max(0, offset - minu_frame)
            end = min(offset + len(seg) + minu_frame, total_duration_frames)
            segments.append(Segment(start, end))
        offset += len(seg)

    return segments


def strm(
    probs: np.array,
    max_segment_length: float = 18,
    min_segment_length: float = 0.2,
    min_pause_length: float = 0.2,
    threshold: float = 0.5,
) -> list[Segment]:
    frame_length = WAV2VEC_FRAME_LEN / 1000
    max_segm_len_steps = int(max_segment_length / frame_length)
    min_segm_len_steps = int(min_segment_length / frame_length)
    min_pause_len_steps = int(min_pause_length / frame_length)

    preds = "".join((probs > threshold).astype(np.int).astype(str))
    splitted_predictions = split_predictions_strm(
        preds,
        max_segm_len_steps,
        min_segm_len_steps,
        min_pause_len_steps,
    )
    segments = get_segments(
        splitted_predictions,
        frame_length,
    )

    return segments


def is_pause(x: str) -> bool:
    return (set(x) == set("0")) or (x == "")


def get_pauses(pred: str) -> list[str]:
    return re.findall(r"0{1,}", pred)


def split_predictions_strm(
    preds: str, max_segm_len: int, min_segm_len: int, min_pause_len: int
) -> list[str]:
    """
    Implementation of the "Streaming" segmentation algorithm of Gaido et al, 2021
    The pause predictions are done before-hand but they are loaded in a
    streaming fashion to simulate the scenario of an audio stream
    Args:
        preds (str): binary predictions for the audio
        max_segm_len (int): maximum allowed segment length
        min_segm_len (int): minimum allowed segment length
        min_pause_len (int): minimum length of a pause
    Returns:
        list[str]: splitted binary predictions
    """

    total_duration_frames = len(preds)
    start = 0
    leftover = ""
    splitted_preds = []

    while start < total_duration_frames:
        end = min(start + max_segm_len - len(leftover), total_duration_frames)
        current_pred = leftover + preds[start:end]

        first_part = current_pred[:min_segm_len]
        second_part = current_pred[min_segm_len:]

        # find continuous pause patterns
        pauses = get_pauses(second_part)

        # get max pause
        pauses.sort(key=lambda s: len(s))
        max_pause = pauses[-1] if len(pauses) else ""

        if len(max_pause) > min_pause_len:
            first_part_b, leftover = second_part.split(max_pause, maxsplit=1)
            if is_pause(first_part):
                splitted_preds.append(first_part)
                if len(first_part_b):
                    splitted_preds.append(first_part_b)
            else:
                splitted_preds.append(first_part + first_part_b)
            splitted_preds.append(max_pause)

        else:
            splitted_preds.append(current_pred)
            leftover = ""

        start = end

    return splitted_preds


def moving_average(
    arr: np.array,
    window: int,
) -> np.array:
    moving_averages = []

    i = 0
    while i < len(arr):
        part_arr = arr[max(0, i - window + 1) : i + 1]
        window_average = sum(part_arr) / len(part_arr)

        moving_averages.append(window_average)
        i += 1

    return np.array(moving_averages)


def pthr(
    probs: np.array,
    max_segment_length: float = 18,
    min_segment_length: float = 0.2,
    max_lerp_range: float = 0,
    min_lerp_range: float = 0,
    threshold: float = 0.5,
    moving_average_window: float = 0,
) -> list[Segment]:
    frame_length = WAV2VEC_FRAME_LEN / 1000
    max_segm_len_steps = int(max_segment_length / frame_length)
    min_segm_len_steps = int(min_segment_length / frame_length)
    max_lerp_steps = int(max_lerp_range / frame_length)
    min_lerp_steps = int(min_lerp_range / frame_length)
    max_lerp_left_steps = max_segm_len_steps - max_lerp_steps
    min_lerp_right_steps = min_segm_len_steps + min_lerp_steps

    # filter
    thresholds = np.full((max_segm_len_steps), threshold)
    thresholds[:min_segm_len_steps] = 0

    # lerp
    thresholds[min_segm_len_steps:min_lerp_right_steps] = np.arange(
        min_lerp_steps, dtype=float
    ) / (min_lerp_steps / threshold)
    thresholds[max_lerp_left_steps:max_segm_len_steps] = threshold + np.arange(
        max_lerp_steps, dtype=float
    ) / (max_lerp_steps / threshold)

    # moving average
    if moving_average_window > 0:
        moving_average_window_steps = int(moving_average_window / frame_length)
        probs = moving_average(probs, moving_average_window_steps)

    total_frames = len(probs)
    start = 0

    segments = []

    # expand each segment by a few seconds (0.06 s)
    minu_frame = TARGET_SAMPLE_RATE * 0.06

    start = 0
    while start < total_frames:
        if probs[start] <= threshold:
            start += 1
            continue
        else:
            # simulating online decoding
            # need to change here for online application
            part_probs = probs[start : start + len(thresholds)]
            end_cands = part_probs <= thresholds[: len(part_probs)]
            end_cands_idx = np.where(end_cands)[0]

            if len(end_cands_idx) > 0:
                end = start + end_cands_idx[0]
            else:
                end = min(start + len(thresholds), total_frames - 1)

            segments.append(
                Segment(
                    max(0, start - minu_frame), min(end + minu_frame, total_frames - 1)
                )
            )

            start = end + 1

    return segments


def update_yaml_content(
    yaml_content: list[dict], segments: list[Segment], wav_name: str
) -> list[dict]:
    """extends the yaml content with the segmentation of this wav file
    Args:
        yaml_content (list[dict]): segmentation in yaml format
        segments (list[Segment]): resulting segmentation from pdac
        wav_name (str): name of the wav file
    Returns:
        list[dict]: extended segmentation in yaml format
    """
    for sgm in segments:
        yaml_content.append(
            {
                "duration": sgm.duration,
                "offset": sgm.offset,
                "rW": 0,
                "uW": 0,
                "speaker_id": "NA",
                "wav": wav_name,
            }
        )

    return yaml_content


def update_tree_yaml_content(
    yaml_content: list[dict],
    tree: list[Segment],
    wav_name: str,
    max_segment_length: float,
    min_segment_length: float,
) -> list[dict]:
    """extends the yaml content with the segmentation tree of this wav file
    Args:
        yaml_content (list[dict]): segmentation in yaml format
        segments (list[Segment]): resulting segmentation from pdac
        wav_name (str): name of the wav file
    Returns:
        list[dict]: extended segmentation in yaml format
    """
    for i, sgm in enumerate(tree):
        if sgm.duration > max_segment_length or sgm.duration < min_segment_length:
            continue
        yaml_content.append(
            {
                "duration": sgm.duration,
                "offset": sgm.offset,
                "rW": 0,
                "uW": 0,
                "speaker_id": str(i),
                "wav": wav_name,
            }
        )

    return yaml_content
