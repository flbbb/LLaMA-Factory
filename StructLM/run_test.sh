CFG_PREFIX=$1
INP_MAX_LEN=${2:-4096}
EVAL_BSIZE=${3:-64}

kwargs=" 
--overwrite_output_dir \
--per_device_eval_batch_size ${EVAL_BSIZE} \
--generation_max_length 1024 \
"

echo ""
echo "STARTING EVALUATION FOR: ${CFG_PREFIX}"
echo ""

# # if there is already a file named predictions_predict.json in the folder, say that
# if [ -f "output/${CFG_PREFIX}_${DATASET_NAME}/predictions_predict.json" ]; then
#     echo "output/${CFG_PREFIX}_${DATASET_NAME}/predictions_predict.json already exists, skipping"
#     exit 0
# fi
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

python StructLM/eval_json.py --run_name ${CFG_PREFIX}
