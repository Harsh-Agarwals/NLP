import zipfile
from pathlib import Path
import requests
import os

def get_data(data_path="../data/food-101", folder="pizza_steak_sushi.zip", api="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"):
    data_path = data_path

    if os.path.exists(data_path) == True:
        print("Path already exists")
    else:
        os.mkdir(data_path)
        with open(os.path.join(data_path, folder), "wb") as f:
            request = requests.get(api)
            print('getting data...')
            f.write(request.content)

        with zipfile.ZipFile(os.path.join(data_path, folder), 'r') as zip_ref:
            zip_ref.extractall(data_path)
            print("Unzipped data file...")
