set -ex
python attack.py --name raw2recap --model pix2pix --direction AtoA --dataset_mode unaligned --norm instance --gpu_ids 1 --serial_batches --eval --preprocess exp 