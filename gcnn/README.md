Toy Keras example using Gated Convoluational Networks for language model on IMDB datasets.

Note:
Only work for tensorflow backend, see [issue 1](https://github.com/DingKe/nn_playground/issues/1).

## Prepare data
change current directory to  data, and follow the instuctions in imdb_preprocess_semi.py.

## Run

### Word LM
python imdb_lm_gcnn.py

### Charactor LM
python char_lm_gcc.py


## References
* Dauphin et al. [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083).
