#!/bin/bash

. $(dirname $0)/path.sh || exit 1;

# joint-s2t-mustc-en-de
en_de_model_path=${ST_MODELS_PATH}/joint-s2t-mustc-en-de
mkdir -p $en_de_model_path
for file in {checkpoint_ave_10.pt,config.yaml,src_dict.txt,dict.txt,spm.model}; do
  wget https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/${file} -O $en_de_model_path/${file}
done

sed -i "s#spm.model#${en_de_model_path}\/spm.model#g" ${en_de_model_path}/config.yaml
sed -i "s# dict.txt# ${en_de_model_path}\/dict.txt#g" ${en_de_model_path}/config.yaml
sed -i "s#src_dict.txt#${en_de_model_path}\/src_dict.txt#g" ${en_de_model_path}/config.yaml
python3 ${SHAS_ROOT}/src/data_prep/fix_joint_s2t_cfg.py -c ${en_de_model_path}/checkpoint_ave_10.pt
