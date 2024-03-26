# decoderfromscratch

Simple probabilistic generative Transformer.
The code is inspired by the book of Huggingface et the annotatedTransformer website.

## To train a model
Just clone this repository, and from the directory decoderfromscratch execute the training script:

python3 train_transformer.py

For the defaut settings you need a GPU with 32 gigas. Think to lower the batch size and/or the block size for smaller GPU.

## To generate some text

A pretrained model is in the checkpoint directory. Just execute the generation script:

python3 generate.py

