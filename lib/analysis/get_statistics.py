import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.append(str(Path(__file__).parents[1]))
from eval_scripts.score import (
    get_parallel,
    score_sentence_bertscore,
    score_sentence_bleu_p1,
)


def main():
    working_dir = Path(sys.argv[1])
    lang = sys.argv[2]
    hyp = working_dir / "__translation"
    ref = working_dir / "__mreference"
    yaml_path = working_dir / "custom_segments.yaml"

    mwer_segmenter_root = Path(os.getcwd()) / "tools/mwerSegmenter"

    results_dir = working_dir / "statistics"
    os.makedirs(results_dir, exist_ok=True)
    os.chdir(results_dir)
    print(f"results path: {os.getcwd()}")

    # align the references with the hypotheses
    mwersegment_cmd = (
        f"{mwer_segmenter_root}/mwerSegmenter" f" -mref {hyp} -hypfile {ref} -usecase 1"
    )
    subprocess.run(mwersegment_cmd, shell=True)

    # calculate bleu per sentence
    bleu = score_sentence_bleu_p1(
        results_dir / "__segments",
        hyp,
        results_dir / "scores.sentence.bleu",
    )

    # calculate bertscore per sentence
    p, r, f1 = score_sentence_bertscore(
        results_dir / "__segments",
        hyp,
        results_dir / "scores.sentence.bertscore",
        lang,
    )

    # generate tsv
    col = []
    with open(yaml_path, "r") as yaml_f:
        segmentation = yaml.load(yaml_f, Loader=yaml.BaseLoader)
    col.append(["Duration"])
    for seg in segmentation:
        col[0].append(seg["duration"])

    ref_l, hyp_l = get_parallel(results_dir / "__segments", hyp)

    col.append(["Hyp"] + hyp_l)
    col.append(["Ref"] + ref_l)
    col.append(["BLEU"] + bleu)
    col.append(["BERTScore(P)"] + p)
    col.append(["BERTScore(R)"] + r)
    col.append(["BERTScore(F1)"] + f1)

    col = list(np.array(col).T)

    with open(
        results_dir / "sentence_statistics.tsv", mode="w", newline="", encoding="utf-8"
    ) as f:
        tsv_writer = csv.writer(f, delimiter="\t")
        tsv_writer.writerows(col)


if __name__ == "__main__":
    main()
