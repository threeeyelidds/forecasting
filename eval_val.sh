export PYTHONPATH=$PWD:$PYTHONPATH

python tools/short_term_anticipation/evaluate_short_term_anticipation_results.py \
    short_term_anticipation/results/short_term_anticipation_results_val.json \
    /mnt/vol_c/ego4d_data/v1/annotations/fho_sta_val.json
