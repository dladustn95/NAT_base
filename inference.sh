#for var in $(seq 0 1 8);
#do
#  CUDA_VISIBLE_DEVICES=1 python -W ignore inference.py --hparams logs/tb_logs/default/version_4/hparams.yaml --model_binary logs/model_chp2/epoch\=0$var.ckpt --testfile data/test.src
#done

for var in $(seq 1 1 1);
do
  CUDA_VISIBLE_DEVICES=0 python -W ignore inference.py \
  --hparams logs/tb_logs/default/version_3/hparams.yaml \
  --model_binary logs/version3/epoch=49-val_loss=16.552.ckpt \
  --testfile data/iwslt14/test --outputfile test.txt --length_beam_size 5
done
