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
Then download the required data and models from [this link](https://zenodo.org/records/14736478?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQzOGExZGFmLTY2ZmYtNGNmYS05YTI1LWI5ZjA2N2E4N2I4NiIsImRhdGEiOnt9LCJyYW5kb20iOiJmMDUwMDg2NzBkYzBiMTJiYTM0MDVmM2ExODFjN2RjNSJ9.63km8trNlAK4djWk4r7nHbOYfbjPM9wWiNa-0RNmv1dOKuz-dvzb1WFAtxh2E_6w9lgLEa4Ltq5EHX22557dlQ)

## Test Recapture Detection
In the recapture-detection directory, run the following command:
```bash
python main.py --test --test_path TEST_PATH --config CONFIG --test_raw_dirnames RAW_DIRNAMES --test_recap_dirnames RECAP_DIRNAMES
```
where `TEST_PATH` is the path to the model, `CONFIG` is the path to the configuration file, `RAW_DIRNAMES` is the list of raw directory names, and `RECAP_DIRNAMES` is the list of recapture directory names. Ensure that the configuration file matches the model being tested. Results should be printed to the terminal.


## Test Deepfake Detection
In the deepfake-detection directory, first update dataset paths in `dataset_paths.py` and move the deepfake detection model weights are in the `pretrained_weights` folder. 

The following command:
```bash
./test.sh
```
will run all the deepfake detection tests at once and save the results in the `deepfake-detection/results` directory.


## Training Chimera
To train Chimera, you must collect data using a fixed screen and camera setup. Then, use the pix2pix training script to train the model. The training script is located in the `pytorch-CycleGAN-and-pix2pix` directory. 

Examples of training commands are provided in the `pytorch-CycleGAN-and-pix2pix/scripts/train_pix2pix.sh` file; ideal parameters depend on setup. Refer to the paper for more details on the training process.