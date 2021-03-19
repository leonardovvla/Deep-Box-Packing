python main.py \
    --use_cuda=False \
    --activity=training \
    \
    --information_window=10 \
    --total_obj_num=150 \
    --min_size=1 \
    --max_size=4 \
    --container_width=8 \
    --container_length=6 \
    --container_height=6 \
    \
    --epoch_num=200 \
    --valid_size=10 \
    --train_size=320 \
    --batch_size=32 \
    --embed_size=32 \
    --actor_lr=5e-4 \
    \
    --num_layers=2 \
    --forward_expansion=1 \
    --heads=1 \
    \
    --stable_placements=True \
    --access=True \
    \
    --title=DBP-Thesis-Train \
    \
    --checkpoint=./results/10/DBP-Thesis-Stable_True-training-150-320-32-32-2-1-1-Bsparse-Vfin-0.0005-RTG-SL-2021-02-27-15-33
