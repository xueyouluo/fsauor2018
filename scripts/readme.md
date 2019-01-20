## Train Model from scratch with GPU

This is a simple script used to run this code from scratch.

> The data preprocess steps are no the same as the one I used during the competition, so the final f1 score may be not the same:

Some major differences:
- Jieba is used instead of LTP here
- NER is not used here
- No custom dictionary used here

### 1. Download raw data

- Download data from this [data link](https://drive.google.com/file/d/1OInXRx_OmIJgK3ZdoFZnmqUi0rGfOaQo/view?usp=sharing).
- Unzip the files to get the raw csv files.

### 2. Download pretrained embedding file

- Download embedding file from this [embedding link](https://pan.baidu.com/s/1tUghuTno5yOvOx4LXA9-wg).
- Unzip the file to get the embedding file.

### 3. Preprocess data

Modify the file paths in preprocess.sh:
- TRAIN_FILE
- VALIDATION_FILE
- TESTA_FILE
- TESTB_FILE
> Refer to 1 to get the data file path
- EMBEDDING_FILE
> Refer to 2 to get the embedding file path
- VOCAB_SIZE
> You can try different vocab size

Then run

```
bash preprocess.sh
```

We will create all the files needed under ./data folder.

### 4. Run training

Change your workdir to parent folder, and run the training scripts:

```
bash bash/elmo_train.sh
```

### 5. Run inference

After training, we can get the predicted results of test files:

```
bash bash/elmo_inference.sh
```