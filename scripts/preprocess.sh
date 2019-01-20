#!/bin/bash

# Modify the following values depend on your environment
# Path to the csv files
TRAIN_FILE=/data/xueyou/data/ai_challenger_sentiment/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv
VALIDATION_FILE=/data/xueyou/data/ai_challenger_sentiment/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv
TESTA_FILE=/data/xueyou/data/ai_challenger_sentiment/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv
TESTB_FILE=/data/xueyou/data/ai_challenger_sentiment/ai_challenger_sentimetn_analysis_testb_20180816/sentiment_analysis_testb.csv

# Path to pretrained embedding file
EMBEDDING_FILE=/data/xueyou/data/embedding/sgns.sogou.word

VOCAB_SIZE=50000

# Create a folder to save training files
mkdir -p data

echo 'Process training file ...'
python data_preprocess.py \
    --data_file=$TRAIN_FILE \
    --output_file=data/train.json \
    --vocab_file=data/vocab.txt \
    --vocab_size=$VOCAB_SIZE

echo 'Process validation file ...'
python data_preprocess.py \
    --data_file=$VALIDATION_FILE \
    --output_file=data/validation.json

echo 'Process testa file ...'
python data_preprocess.py \
    --data_file=$TESTA_FILE \
    --output_file=data/testa.json

# Uncomment following code to get testb file
# echo 'Process testb file ...'
# python data_preprocess.py \
#     --data_file=$TESTB_FILE \
#     --output_file=data/testb.json

echo 'Get pretrained embedding ...'
python data_preprocess.py \
    --data_file=$EMBEDDING_FILE \
    --output_file=data/embedding.txt \
    --vocab_file=data/vocab.txt \
    --embedding=True

echo "Get label file ..."
cp ../labels.txt data/labels.txt