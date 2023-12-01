export PYTHONPATH=$PWD:$PYTHONPATH

python tools/short_term_anticipation/dump_frames_to_customised_formats.py \
    /mnt/vol_c/ego4d_data/v1/annotations/ \
    /mnt/vol_c/ego4d_data/v1/full_scale/ \
    short_term_anticipation/data/lmdb \
    --save_as_film --stride 4 --context_frames 32
    # --save_as_video --stride 1