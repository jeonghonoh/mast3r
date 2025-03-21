

OUTPUT_DIR=./output/dualarm/real/seq_01/
N_VIEW=9
gs_train_iter=1000

GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 demo.py -o ${OUTPUT_DIR} -r 1 --n_views ${N_VIEW} --optim_pose --iterations ${gs_train_iter}