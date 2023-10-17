WORK_DIR="mnist-realdsm"
ROOT="./root/"
SAMPLES="eval_fid"
rm -rf ${ROOT}/${WORK_DIR}_seed_0
python main.py -cc configs/default_mnist.txt  -sc configs/specific_mnist.txt \
    --mode train --cont_nbr 1 --workdir ${WORK_DIR} --n_gpus_per_node 1 \
    --training_batch_size 64 --testing_batch_size 64 --sampling_batch_size 64 \
    --dataset mnist --root ${ROOT}
