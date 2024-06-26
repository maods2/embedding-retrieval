import os
import re
import pickle

base_path = "./embeddings"

# walk through all files of the dir
for root, subdirs, files in os.walk(base_path):
    for trainfile in files:
        if not trainfile.endswith("_train.pickle"):
            continue
        testfile = re.sub("_train", "_test", trainfile)
        with open(os.path.join(root, trainfile), "rb") as f:
            train_data = pickle.load(f)
        with open(os.path.join(root, testfile), "rb") as f:
            test_data = pickle.load(f)
        # assert the same order of classes
        print(f"Testing for dataset {trainfile}")
        try:
            assert dict.fromkeys(train_data["classes"]) == dict.fromkeys(test_data["classes"])
            print("OK")
        except AssertionError:
            print("ERROR")
            print(f"{trainfile}: {dict.fromkeys(train_data['classes'])}")
            print(f"{testfile}: {dict.fromkeys(test_data['classes'])}")