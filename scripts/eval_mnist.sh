WORK_DIR="mnist"
ROOT="./root"
SAMPLES="eval_fid"
python main.py -cc configs/default_mnist.txt  -sc configs/specific_mnist.txt \
    --mode eval --workdir ${WORK_DIR} --n_gpus_per_node 1 \
    --training_batch_size 64 --testing_batch_size 64 --sampling_batch_size 64 \
    --dataset mnist --root ${ROOT} \
    --eval_folder ${SAMPLES} --eval_sample 
# mv ${ROOT}/${WORK_DIR}_seed_0/${SAMPLES}/samples/* ${ROOT}/gmm_samples/