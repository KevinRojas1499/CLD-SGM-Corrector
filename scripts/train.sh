WORK_DIR="swiss_roll"
ROOT="./root"
SAMPLES="eval_fid"
# rm -rf ${ROOT}/${WORK_DIR}_seed_0
python main.py -cc configs/default_toy_data.txt  --mode train --cont_nbr 1 --workdir ${WORK_DIR} --n_gpus_per_node 1 \
    --training_batch_size 512 --testing_batch_size 512 --sampling_batch_size 512 \
    --dataset multimodal_swissroll --root ${ROOT}