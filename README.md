# RythmCap
RythmCap uses deep learning to mimic the music captioning style of [Prof. Paolo Bientinesi](http://hpac.rwth-aachen.de/~pauldj). This project was carried out as a part of my Master thesis - ["Analyses of CNNs for automatic tagging of music tracks"](https://drive.google.com/file/d/1ljEz-KJPpg3KZHLHEh891Tl3DNh2V1Lp/view?usp=sharing). Framework used for training CNNs is available in [here](https://github.com/as641651/CNNs-for-automatic-tagging-of-music-tracks)

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
