# Quick start for ViGA+DEA

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
