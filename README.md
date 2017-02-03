# RythmCap
This repository contains the algorthim that tries to mimic the music captioning style of [Prof. Paolo Bientinesi](http://hpac.rwth-aachen.de/~pauldj). (Still under construction)

## Clone this repository

```bash
git clone https://github.com/as641651/RythmCap.git
```

## Install Dependencies

RythmCap is implemented in [Torch](http://torch.ch/)

```bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
```

After installing torch, you can install / update these dependencies by running the following:

```bash
luarocks install rnn
luarocks install hdf5
```
For GPU acceleration (Optional)

```bash
luarocks install cutorch
luarocks install cunn
```
Python dependencies:
[librosa](https://librosa.github.io/librosa/install.html)
[h5py](http://docs.h5py.org/en/latest/build.html)

## Usage

```bash
th caption.lua -i <path_to_input_track> -m <path_to_model>
```

To see the tags used by the model, use the option '-v'
```bash
th caption.lua -m <path_to_model> -v
```

[Download Sample model](https://drive.google.com/open?id=0B8Uc-OssxXlDV1FheU1BVEdkVVE) [~145MB]
