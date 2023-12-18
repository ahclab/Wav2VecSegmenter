export ROOT=$(cd $(dirname $0)/..; pwd)

# tools
export TOOL_PATH=${ROOT}/tools
export SHAS_ROOT=${TOOL_PATH}/SHAS
export MWERSEGMENTER_ROOT=${TOOL_PATH}/mwerSegmenter
export FAIRSEQ_ROOT=${TOOL_PATH}/fairseq

# data
export MUSTC_ROOT=${ROOT}/data/corpus/MuST-C/v2.0_IWSLT2022
export EUROPARL_ROOT=${ROOT}/data/corpus/Europarl-ST/v1.1
export SEGM_DATASETS_ROOT=${ROOT}/data/training

# models
export MODELS_PATH=${ROOT}/models
export ST_MODELS_PATH=${MODELS_PATH}/st
export RESULTS_ROOT=${MODELS_PATH}/segmentation
