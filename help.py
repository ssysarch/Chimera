import json
import os
import shutil

dir = "/home/seongbin/transfer_data/data/iphone/stylegan2D"
json_file_path = "/home/seongbin/transfer_data/data/mytest.json"
newdir = "/home/seongbin/transfer_data/data/iphone/recap_monitor"

with open(json_file_path, "r") as f:
    json_files = json.load(f)

    for file in json_files:
        src_path = os.path.join(dir, file)
        dst_path = os.path.join(newdir, file)
        # make directory if not exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)