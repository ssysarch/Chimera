# Chimera

## Setup and Installation
First create a conda environment using the following command:
```bash
conda env create -f environment.yml
```
Then activate the environment using the following command:
```bash
conda activate chimera
```
Then download the required data and models from the following links:

## Test Recapture Detection
In the recapture-detection directory, run the following command:
```bash
python main.py --test --test_path TEST_PATH --config CONFIG --test_raw_dirnames RAW_DIRNAMES --test_recap_dirnames RECAP_DIRNAMES
```
where `TEST_PATH` is the path to the test dataset, `CONFIG` is the path to the configuration file, `RAW_DIRNAMES` is the list of raw directory names, and `RECAP_DIRNAMES` is the list of recapture directory names.


## Test Deepfake Detection
In the deepfake-detection directory, first update dataset paths in `dataset_paths.py` and ensure that the deepfake detection model weights are in the `pretrained_weights` folder. 

The following command:
```bash
./test.sh
```
will run all the deepfake detection tests at once and save the results in the `results` directory.
