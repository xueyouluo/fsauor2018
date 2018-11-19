python main.py \
--mode=inference \
--data_files=/data/xueyou/data/ai_challenger_sentiment/v3/data/test.json \
--label_file=./labels.txt \
--vocab_file=/data/xueyou/data/ai_challenger_sentiment/v3/data/vocab.txt \
--out_file=/data/xueyou/data/ai_challenger_sentiment/sprint/out.json \
--prob=False \
--batch_size=300 \
--feature_num=20 \
--checkpoint_dir=/data/xueyou/data/ai_challenger_sentiment/sprint/elmo_ema_1116