CUDA_VISIBLE_DEVICES=0,1 python train_nat.py \
  --max_epochs 50 --warmup_ratio 0.05 \
  --batch_size 768 --max_len 64 \
  --num_workers 8 --lr 5e-4 \
  --default_root_dir logs  --gpus 2 \
  --train_file data/iwslt14/train --valid_file data/iwslt14/valid \
  --src_lang de --tgt_lang en