CUDA_VISIBLE_DEVICES=0 python -W ignore inference.py \
  --hparams logs/tb_logs/default/version_3/hparams.yaml \
  --model_binary logs/version3/epoch=49-val_loss=16.552.ckpt \
  --testfile data/iwslt14/test --outputfile test.txt --length_beam_size 5 \
  --src_lang de --tgt_lang en
