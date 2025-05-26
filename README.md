# Reverse Distribution based Video Moment Retrieval for Effective Bias Elimination

This is implementation for the paper "Reverse Distribution based Video Moment Retrieval for Effective Bias Elimination".
[[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/33302/35457)

## Glove download


Download glove.840B.300d from https://nlp.stanford.edu/data/glove.840B.300d.zip.


## Quick start for ViGA+DEA

**Preparing dataset**

1. Charades features is downloaded from [this page](https://app.box.com/s/nruly3nocdzid9atm78n12i7jbvu8bh4/folder/158103085528).
Activitynet features is downloaded from [this page](http://activity-net.org/challenges/2016/download.html). We extracted the features from sub_activitynet_v1-3.c3d.hdf5 as individual `.npy` files.

2. Change the feature path in `config.yaml` to your data storage path.

**Preparing environment**

```shell script
$ conda create --name viga python=3.8
$ source activate viga
(viga)$ conda install pytorch=2.4.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
(viga)$ pip install numpy scipy pyyaml tqdm matplotlib
```

**Train** and **Test**

```shell script
# train ViGA+DEA on Charades-STA dataset in Traditional VMR 
python -m src.experiment.train --task charadessta --config-file config.yaml
# train ViGA+DEA on Charades-STA dataset in Resplitting VMR
python -m src.experiment.train --task Charades_CD --config-file config.yaml
# train ViGA+DEA on Charades-STA dataset in Redis VMR 
python -m src.experiment.biases_train --task charadessta --config-file config.yaml

# train ViGA+DEA on ActivityNet Captions dataset in Traditional VMR 
python -m src.experiment.train --task activitynetcaptions --config-file config.yaml
# train ViGA+DEA on ActivityNet Captions dataset in Resplitting VMR
python -m src.experiment.train --task ANet_CD --config-file config.yaml
# train ViGA+DEA on ActivityNet Captions dataset in Redis VMR 
python -m src.experiment.biases_train --task activitynetcaptions --config-file config.yaml

```

## Quick start for VSLNet+DEA

**Preparing dataset**
1. Downloaded from [this page](https://app.box.com/s/nruly3nocdzid9atm78n12i7jbvu8bh4/folder/158103085528).

2. Put features into `./data/features` using soft link.
```shell script
mkdir -p ./data/features/charades
ln -s /path/to/charades_features ./data/features/charades/new
mkdir -p ./data/features/charades_cd
ln -s /path/to/charades_features ./data/features/charades_cd/new
mkdir -p ./data/features/activitynet
ln -s /path/to/activitynet_features ./data/features/activitynet/new
mkdir -p ./data/features/activitynet_cd
ln -s /path/to/activitynet_features ./data/features/activitynet_cd/new
ln -s /path/to/glove.840B.300d.txt ./data/features/glove.840B.300d.txt
```   

**Preparing environment**

```shell script
conda create --name vslnet_dea python=3.8
conda activate vslnet_dea
conda install pytorch=2.4.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c anaconda nltk pillow=6.2.1
conda install -c conda-forge transformers opencv moviepy tqdm youtube-dl
python -m nltk.downloader punkt
```

**Train** and **Test**
```shell script
# train VSLNet+DEA on Charades-STA dataset in Traditional VMR 
python main_t7.py --task charades --predictor rnn --mode train --settings traditional
# train VSLNet+DEA on Charades-STA dataset in Resplitting VMR
python main_t7.py --task charades_cd --predictor rnn --mode train --settings traditional
# train VSLNet+DEA on Charades-STA dataset in Redis VMR 
python main_t7.py --task charades --predictor rnn --mode train 

# train VSLNet+DEA on ActivityNet Captions dataset in Traditional VMR 
python main_t7.py --task activitynet --predictor rnn --mode train --settings traditional
# train VSLNet+DEA on ActivityNet Captions dataset in Resplitting VMR
python main_t7.py --task activitynet_cd --predictor rnn --mode train --settings traditional
# train VSLNet+DEA on ActivityNet Captions dataset in Redis VMR 
python main_t7.py --task activitynet --predictor rnn --mode train 

```

