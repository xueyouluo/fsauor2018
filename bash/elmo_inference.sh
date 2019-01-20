python main.py \
--mode=inference \
--data_files=scripts/data/testa.json \
--label_file=scripts/data/labels.txt \
--vocab_file=scripts/data/vocab.txt \
--out_file=scripts/data/out.testa.json \
--prob=False \
--batch_size=300 \
--feature_num=20 \
--checkpoint_dir=scripts/data/elmo_ema_0120