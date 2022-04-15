import os
import tarfile
import torch

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def extract_tgz(tar_url, extract_path):
    tar = tarfile.open(tar_url, 'r')
    for item in tar:
        tar.extract(item, extract_path)

def create_raw_data(path, target):
    with open(path) as f:
        text = f.readlines()
        for line in text:
            # if line != '\n' and line != '.\n':
            target.append(line.lower())
    return target

