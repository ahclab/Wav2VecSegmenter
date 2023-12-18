#!/bin/bash

. $(dirname $0)/path.sh || exit 1;

git clone https://github.com/mt-upc/SHAS.git $SHAS_ROOT

mkdir -p $MWERSEGMENTER_ROOT
wget --no-check-certificate https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz
tar -zxvf mwerSegmenter.tar.gz -C ${MWERSEGMENTER_ROOT} --strip-components 1
rm -r mwerSegmenter.tar.gz
patch ${MWERSEGMENTER_ROOT}/segmentBasedOnMWER.sh ${ROOT}/scripts/patch/segmentBasedOnMWER.patch

git clone -b shas https://github.com/mt-upc/fairseq.git $FAIRSEQ_ROOT
sed -i.bak 's/hydra-core>=1.0.7,<1.1/hydra-core==1.1.1/g' ${FAIRSEQ_ROOT}/setup.py
sed -i 's/omegaconf<2.1/omegaconf==2.1/g' ${FAIRSEQ_ROOT}/setup.py
pip install --editable $FAIRSEQ_ROOT
