export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p short_term_anticipation/results
python scripts/run_sta.py \
    --cfg configs/Ego4dShortTermAnticipation/SLOWFAST_32x1_8x4_R50.yaml \
    TRAIN.ENABLE False TEST.ENABLE True ENABLE_LOGGING False \
    CHECKPOINT_FILE_PATH /mnt/vol_c/ego4d_data/v1/sta_models/slowfast_model.ckpt \
    RESULTS_JSON short_term_anticipation/results/short_term_anticipation_results_val.json \
    CHECKPOINT_LOAD_MODEL_HEAD True \
    DATA.CHECKPOINT_MODULE_FILE_PATH "" \
    CHECKPOINT_VERSION "" \
    TEST.BATCH_SIZE 1 NUM_GPUS 1 \
    EGO4D_STA.OBJ_DETECTIONS /mnt/vol_c/ego4d_data/v1/sta_models/object_detections.json \
    EGO4D_STA.ANNOTATION_DIR /mnt/vol_c/ego4d_data/v1/annotations \
    EGO4D_STA.RGB_LMDB_DIR short_term_anticipation/data/lmdb/ \
    EGO4D_STA.TEST_LISTS "['fho_sta_val.json']"