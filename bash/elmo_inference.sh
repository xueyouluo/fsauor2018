python main.py \
--mode=inference \
--data_files=/data/xueyou/data/corpus/datagrand/test.txt \
--label_file=labels.txt \
--batch_size=128 \
--vocab_file=/data/xueyou/data/corpus/datagrand/xlnet/vocab.txt \
--out_file=/data/xueyou/data/corpus/datagrand/test/out.test.compose_2.txt \
--checkpoint_dir=/data/xueyou/data/corpus/datagrand/test/elmo_mlm_compose