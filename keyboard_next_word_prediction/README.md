## Dataset

Download using the below command
```bash
python
>>> from helpers import download_data
>>> download_data()
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
```
