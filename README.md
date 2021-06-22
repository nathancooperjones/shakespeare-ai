# Shakespeare AI

![](https://images.theconversation.com/files/214570/original/file-20180412-570-18x3skh.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=754&fit=clip)

> "Unpregnant o'ercharged falsely, I'll not, I have been the sourest-natured of my life. I will have I, and he shall have a man. I will not be. Shepherd, [ reads lear and queen of norfolk ] o lord!"
>
> -- *Original Shakespeare from a RNN Model*

> PRINCESS
> I must get you out of there.
>
> SHERLOCK
> Your Majesty, go not thus. This will be
> Your sword, which in my hands I am more
> For the life of this land than to fight with you.
> The Lord of the Manor to keep his vow
> Makes me fear the worst. What sayst thou,
> Hast thou not seen this for yourself?
> The Prince hath, the Jew of the East
> In the Tower of London's Tower. 'Tis true.
>
> PRINCESS
> O, I love him, but my love is not with
> Thou, nor with thyself, nor with thy soul.
> If I could be as tender as he was,
> I would swear all's as fair as he was.
>
> SHERLOCK
> I do not care to be loved by thee.
>
> PRINCESS  Ay, I would wish it were so.
>
> -- *Original Shakespeare from a Fine-Tuned GPT-2 Model*

"Well, it's no Shakespeare," - taken to a whole new level. Generate original Shakespearean text using deep learning, either by training a word-level RNN from scratch or by fine-tuning a smaller version of OpenAI's GPT-2 model that has been trained on practically the entire internet.

-----

### Quick Start
With Python installed and this repository cloned, type the following into the bash shell:
```bash
pip install -r requirements.txt & pip install -r requirements-dev.txt
```

To train a RNN using all Shakespearean texts combined into a single file (included in this repo):
```python
from shakespeare_ai.learner import ShakespeareLearner

learner = ShakespeareLearner(train_file='../data/all-shakespeare.txt')
learner.train()

learner.predict('Costliness')
> "Costliness -- i 'll o'erlook a gentler and alleged of a good? What is 't is? Menas and the approaching grass of his sake? I have been a bachelor, i will have a man. I have given him. [ he draws. Enter a messenger and..."
```

To fine-tune a GPT-2 model using all Shakespearean texts with the special `<|endoftext|>` token (also included in this repo), use the bash shell to:
```bash
# We start by downloading a model needed for GPT-2 to run properly.
# I am using the 345M, but the 117M can also be used instead!
python shakespeare_ai/externals/gpt2/download_model.py 345M

# By this point, you should now have a `models/` directory.
# Now we need to encode all of the texts into a single file ready for GPT-2 consumption.
python shakespeare_ai/externals/gpt2/encode.py --model_name 345M data/texts/ data/encoded_shakespeare_texts.npz

# Now you can train the model for however many iterations you see fit. Cancel at any point using *Ctrl-C*
# and the model will save before cancelling.
python shakespeare_ai/externals/gpt2/train.py --dataset data/encoded_shakespeare_texts.npz --model_name 345M

# And now, you have `checkpoint` and `samples` directories!
# Let's move files over to another directory so we can make predictions.
mkdir models/shakespeare
cp checkpoint/run1/checkpoint models/shakespeare/
cp `ls -t checkpoint/run1/model* | head -3` models/shakespeare/
cp models/345M/encoder.json models/shakespeare/
cp models/345M/hparams.json models/shakespeare/
cp models/345M/vocab.bpe models/shakespeare/

# You are now able to run one of the following:
# Generate samples until you cancel with *Ctrl-C*...
python shakespeare_ai/externals/gpt2/src/generate_unconditional_samples.py --model_name shakespeare
# ... or use inputted text as a seed to a generated sample.
python shakespeare_ai/externals/gpt2/src/interactive_conditional_samples.py --model_name shakespeare
```

### Development
#### Word-Level RNN
The bulk of this repo consists of the LSTM model. Despite this repo's name, this model can be used with any text (given it is long enough) to generate anything from books to speeches to the deepest darkest of internal thoughts!

In order to do this, combine all text sources into a single `.txt` file, then use the `shakespeare_ai.learner` class as normal.

The main files to modify for this model are the following:
* **`prepare.py`**: Reads in the single text file, tokenizes the text using the `nltk` library, and gives each vocabulary word a unique integer representation.
* **`model.py`**: Defines the structure of the RNN model used, which consists of an embedding layer for the vocabulary, a potentially multi-layered LSTM, and a final dense layer output.
* **`learner.py`**: The main interface - a wrapper over the model that allows for a simple interface for training, predicting, saving, and loading the model. I have provided extensive documentation into all of the parameters you are able to tune to make a kickass model.

The best way to see this interface in action is by running the Jupyter notebook in `nbs/RNN.ipynb` by running the following in the bash shell:
```bash
jupyter lab --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

#### GPT-2 Fine-Tuning
Most non-Shakespearean customizations to this will be in how the data is prepared. For this model, the data can be left in a single directory with many files, with each text ending with a `<|endoftext|>` token so the GPT-2 model is able to understand the structure of the text better. From there, the exact commands listed in the `Quick Start` above will allow you to train an amazing model.

The files used for this have been directly cloned from [here](https://github.com/nshepperd/gpt-2) and can be found in this repository under `shakespeare_ai.externals.gpt2`.

-----

<sup>Inspiration from [Trung Tran's](https://machinetalk.org/2019/02/08/text-generation-with-pytorch/) and [Michael Sugimura's](https://shoprunnerblog.wordpress.com/2019/10/08/this-dress-doesnt-exist/) excellent blog posts.</sup>
