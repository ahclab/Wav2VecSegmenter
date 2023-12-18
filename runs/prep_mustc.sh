#!/bin/bash

. $(dirname $0)/path.sh || exit 1;

# MuST-C en-de
mkdir -p ${SEGM_DATASETS_ROOT}/MUSTC/en-de
for split in {dev,tst-COMMON,train}; do
  python ${SHAS_ROOT}/src/data_prep/prepare_dataset_for_segmentation.py \
    -y ${MUSTC_ROOT}/en-de/data/${split}/txt/${split}.yaml \
    -w ${MUSTC_ROOT}/en-de/data/${split}/wav \
    -o ${SEGM_DATASETS_ROOT}/MUSTC/en-de
done
