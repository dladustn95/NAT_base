CUDA_VISIBLE_DEVICES=0,1 python train_nat.py  --gradient_clip_val 1.0 \
		--train_file data/iwslt14/train --valid_file data/iwslt14/valid --test_file data/iwslt14/test \
                --max_epochs 50 \
                --default_root_dir logs \
                --gpus 2 \
                --batch_size 768 --max_len 64 \
                --num_workers 8 --lr 5e-4 \
		--warmup_ratio 0.05