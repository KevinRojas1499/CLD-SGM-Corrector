WORK_DIR="mnist-realdsm"
ROOT="./root"
SAMPLES="eval_fid"
OUT_DIR="mnist_samples_corrector"
python main.py -cc configs/default_mnist.txt  -sc configs/specific_mnist.txt \
    --mode eval --workdir ${WORK_DIR} --n_gpus_per_node 1 \
    --training_batch_size 64 --testing_batch_size 64 --sampling_batch_size 64 \
    --dataset mnist --root ${ROOT} \
    --eval_folder ${SAMPLES} --eval_sample 
mkdir -p ${ROOT}/${OUT_DIR}/
mv ${ROOT}/${WORK_DIR}_seed_0/${SAMPLES}/fid/* ${ROOT}/${OUT_DIR}/
rm -rf ${ROOT}/${WORK_DIR}_seed_0/${SAMPLES}