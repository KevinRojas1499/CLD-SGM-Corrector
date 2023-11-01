WORK_DIR="1dgmm"
ROOT="./root"
SAMPLES="eval_fid"
rm -rf ${ROOT}/${WORK_DIR}_seed_0/eval_fid
rm -rf ${ROOT}/trajectory/*.png
python main.py -cc configs/default_1d_gmm.txt  --mode eval --workdir ${WORK_DIR} --n_gpus_per_node 1 \
    --training_batch_size 512 --testing_batch_size 512 --sampling_batch_size 10000 \
    --dataset gmm --root ${ROOT} \
    --eval_folder ${SAMPLES} --eval_sample 
mv ${ROOT}/${WORK_DIR}_seed_0/${SAMPLES}/samples/* ${ROOT}/1d_ggm_samples/