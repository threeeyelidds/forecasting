export PYTHONPATH=$PWD:$PYTHONPATH

python tools/short_term_anticipation/dump_frames_to_lmdb_files.py \
    /mnt/vol_c/ego4d_data/v1/annotations/ \
    /mnt/vol_c/ego4d_data/v1/full_scale/ \
    short_term_anticipation/data/lmdb \
    --save_as_video --stride 8 --context_frames 256
    # --save_as_video --stride 1