python main.py -cc configs/default_toy_data.txt  --mode train --workdir multi_swiss_roll_vpsde --n_gpus_per_node 1 \
    --training_batch_size 512 --testing_batch_size 512 --sampling_batch_size 512 \
    --dataset multimodal_swissroll --root ./root/