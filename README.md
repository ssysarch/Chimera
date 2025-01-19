# Chimera

## Test Recapture Detection
In the recapture-detection directory, run the following command:
```bash
python main.py --test --test_path TEST_PATH --config CONFIG --test_raw_dirnames RAW_DIRNAMES --test_recap_dirnames RECAP_DIRNAMES
```
where `TEST_PATH` is the path to the test dataset, `CONFIG` is the path to the configuration file, `RAW_DIRNAMES` is the list of raw directory names, and `RECAP_DIRNAMES` is the list of recapture directory names.


## Test Deepfake Detection
In the deepfake-detection directory, first update dataset paths in `dataset_paths.py` and then run the following command:
```bash
./test.sh
```
This will run all the deepfake detection tests at once and save the results in the `results` directory.
