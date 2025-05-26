# Quick start for VSLNet+DEA

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

