# Translate english to Vietnamese language
## Install requirement
```bash
$ pip install -r requirements.txt
$
```
## Train modle
- Train from scratch
```bash
$ python process.py --is-train
```
- Pretrain weight
```bash
$ python process.py --is-train --use-weights
```
Can modify hyperparameters lr, batch_size, n_workers and modify transformer's architecture.
## Translation
```bash
$ python process.py --sentence-src 'the sentence which you want to translate'
```
'the sentence which you want to translate' is source sentence input