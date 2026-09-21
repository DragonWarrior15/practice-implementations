# Keyboard Next Word Prediction
We predict the next word for completed words, and complete the word in case of partially typed word.
First stage is to build a trigram model and bench mark it.

## Dataset

Download using the below command
```bash
python
>>> from helpers import download_data
>>> download_data()
```

or

```bash
uv run python -c "from helpers import download_data; download_data()"
```

What does the dataset contain ?
```bash
python
>>> from datasets import load_dataset
>>> ds = datasets.load("data")
>>> ds.keys()
dict_keys(['train', 'validation', 'test'])
>>> ds['train'][1]
{'text': ' = Valkyria Chronicles III = \n'}
>>> type(ds['train'][1]['text'])
<class 'str'>
```

## Trigram Training
```bash
python
>>> from trigram_train import build_trigram_dictionary
>>> build_trigram_dictionary()
```

or

```bash
uv run python -c "from trigram_train import build_trigram_dictionary; build_trigram_dictionary()"
```
