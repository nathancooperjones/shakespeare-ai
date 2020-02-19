#!/bin/bash

# we start by downloading a model needed for GPT-2 to run properly
python shakespeare_ai/externals/gpt2/download_model.py 345M
# by this point, you should now have a `models/` directory
python shakespeare_ai/externals/gpt2/encode.py --model_name 345M data/texts/ data/encoded_shakespeare_texts.npz
# by this point, you will also have created `data/encoded_shakespeare_texts.npz`
python shakespeare_ai/externals/gpt2/train.py --dataset data/encoded_shakespeare_texts.npz --model_name 345M
# and now, you have `checkpoint` and `samples` directories!

# great - let's move some files over now
mkdir models/shakespeare
cp checkpoint/run1/checkpoint models/shakespeare/
cp `ls -t checkpoint/run1/model* | head -3` models/shakespeare/
cp models/345M/encoder.json models/shakespeare/
cp models/345M/hparams.json models/shakespeare/
cp models/345M/vocab.bpe models/shakespeare/

# now you can run one of the following:
python shakespeare_ai/externals/gpt2/src/generate_unconditional_samples.py --model_name shakespeare
# or
python shakespeare_ai/externals/gpt2/src/interactive_conditional_samples.py --model_name shakespeare
