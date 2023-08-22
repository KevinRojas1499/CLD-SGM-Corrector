WORK_DIR="realdsm"
ROOT="./root"
SAMPLES="eval_fid"
rm -rf ${ROOT}/${WORK_DIR}_seed_0/eval_fid
rm -rf ${ROOT}/trajectory/*.png
python main.py -cc configs/default_gmm.txt  --mode eval --workdir ${WORK_DIR} --n_gpus_per_node 1 \
    --training_batch_size 512 --testing_batch_size 512 --sampling_batch_size 5000 \
    --dataset multimodal_swissroll --root ${ROOT} \
    --eval_folder ${SAMPLES} --eval_sample 