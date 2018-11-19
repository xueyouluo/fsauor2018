# fsauor2018

Code for Fine-grained Sentiment Analysis of User Reviews of AI Challenger 2018.

Single model can achieve 0.71 marco-f1 score.

Testa rank: 27

Testb rank: 16

> The final result is achieved by ensemble 10 models by simple voting.

Issues and starts are welcomed!

## Requirements

tensorflow == 1.4.1

## Data preprocess

The data preprocess code is not provided here, I may release it later.

To use this project, you need fowllowing files:

- train.json / validataion.json / testa.json
- vocab.txt
- embedding.txt
- label.txt

### Training files

You need to preprocess the orginal data to json files, each line of the json line should be like fowllowing:

```json
{"id": "0", "content": "吼吼吼 ， 萌 死 人 的 棒棒糖 ， 中 了 大众 点评 的 霸王餐 ， 太 可爱 了 。 一直 就 好奇 这个 棒棒 糖 是 怎么 个 东西 ， 大众 点评 给 了 我 这个 土老 冒 一个 见识 的 机会 。 看 介绍 棒棒 糖 是 用 <place> 糖 做 的 ， 不 会 很 甜 ， 中间 的 照片 是 糯米 的 ， 能 食用 ， 真是 太 高端 大气 上档次 了 ， 还 可以 买 蝴蝶 结扎口 ， 送 人 可以 买 礼盒 。 我 是 先 打 的 卖家 电话 ， 加 了 微信 ， 给 卖家传 的 照片 。 等 了 几 天 ， 卖家 就 告诉 我 可以 取 货 了 ， 去 <place> 那 取 的 。 虽然 连 卖家 的 面 都 没 见到 ， 但是 还是 谢谢 卖家 送 我 这么 可爱 的 东西 ， 太 喜欢 了 ， 这 哪 舍得 吃 啊 。", "location_traffic_convenience": "-2", "location_distance_from_business_district": "-2", "location_easy_to_find": "-2", "service_wait_time": "-2", "service_waiters_attitude": "1", "service_parking_convenience": "-2", "service_serving_speed": "-2", "price_level": "-2", "price_cost_effective": "-2", "price_discount": "1", "environment_decoration": "-2", "environment_noise": "-2", "environment_space": "-2", "environment_cleaness": "-2", "dish_portion": "-2", "dish_taste": "-2", "dish_look": "1", "dish_recommendation": "-2", "others_overall_experience": "1", "others_willing_to_consume_again": "-2"}
```

To be specific:
- content should be tokeninzed words
    - You can use jieba/ltp to do the segmentation
    - Use NER toolkits to replace place and orginaztion to special tokens '\<place>','\<org>'
- other fields are same as the original data files
- for test files, which labels are unknow, you can leave them to be empty string("")

### Vocab file

I choose the top 50k most common words in training file.

The top 3 words are special tokens, which are:
- \<unk>: unknow token
- \<sos>: start of content
- \<eos>: end of content, also used as padding token

### Embedding file

This is a glove-format embedding file, I use [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors) as pretrained embedding file(which is Sogou News word2vec word embedding).

### Label file

All the label names.

## Train

Refer to bash/elmo_train.sh

## Inference

Refer to bash/elmo_inference.sh