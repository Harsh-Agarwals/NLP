
import os

def get_data_details(folder="../data/food-101"):
    os.chdir(folder)
    for folder in os.listdir():
        if folder.endswith(".zip") or folder.endswith(".jpeg"):
            continue
        print(folder)
        os.chdir(folder)

        for directory in os.listdir():
            os.chdir(directory)
            files = len(os.listdir())
            print(directory + ":" + str(files))
            os.chdir("../")
        os.chdir("..")
