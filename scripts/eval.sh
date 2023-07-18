WORK_DIR="multi_swiss_roll"
ROOT="."
SAMPLES="eval_fid"
rm -rf ${ROOT}/${WORK_DIR}_seed_0/eval_fid
python main.py -cc configs/default_toy_data.txt  --mode eval --workdir ${WORK_DIR} --n_gpus_per_node 1 \
    --training_batch_size 512 --testing_batch_size 512 --sampling_batch_size 512 \
    --dataset multimodal_swissroll --root ${ROOT} \
    --eval_folder ${SAMPLES} --eval_sample 
mv ${ROOT}/${WORK_DIR}_seed_0/${SAMPLES}/samples/* ${ROOT}/comparisons_both/