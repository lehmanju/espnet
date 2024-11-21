#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh

cmd=run.pl
nj=1 # number of GPUs to extract features
stage=1
stop_stage=4

log "$0 $*"
. utils/parse_options.sh

if [ -z "${LRS3}" ]; then
    log "Fill the value of 'LRS3' of db.sh"
    log "Dataset can be download from https://mmai.io/datasets/lip_reading/"
    exit 1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # Extract audio visual features and fuse the two features using pre-trained AV-HuBERT front-ends
    if python -c "import skvideo, skimage, cv2, python_speech_features" &> /dev/null; then
        echo "requirements installed"
    else
        echo "please install required packages by run 'cd ../../../tools; source activate_python.sh; installers/install_visual.sh;'"
        exit 1;
    fi

    tempfile=data/temp
    trap 'rm -rf $tempfile' EXIT
    for dataset in train val test; do
        if [ -e data/${dataset}/feats.scp ]; then
            continue
        fi

        echo "extracting visual feature for [${dataset}]"
        log_dir=data/${dataset}/split_${nj}
        split_scps=""
        mkdir -p ${log_dir}
        for n in $(seq $nj); do
            split_scps="$split_scps ${log_dir}/video.$n.scp"
        done
        ./utils/split_scp.pl data/${dataset}/video.scp $split_scps || exit 1

        ${cmd} JOB=1:$nj ${log_dir}/extract_av_feature.JOB.log python ./local/extract_av_feature.py \
            --file_list ${log_dir}/video.JOB.scp \
            --model ${model_conf} \
            --gpu JOB \
            --write_num_frames ark,t:${log_dir}/num_frames.JOB.txt \
            ark,scp:${log_dir}/feature.JOB.ark,${log_dir}/feature.JOB.scp || exit 1

        for n in $(seq $nj); do
            cat ${log_dir}/feature.${n}.scp
        done > data/${dataset}/feats.scp

        for n in $(seq $nj); do
            cat ${log_dir}/num_frames.${n}.txt
        done > data/${dataset}/num_frames.txt
    done

    for dataset in train val test; do
        utils/fix_data_dir.sh data/${dataset}
    done

fi
