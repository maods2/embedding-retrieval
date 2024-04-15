# Retrieval Embeddings

## In order to generate embeddings based on image data set is necesary to run the scripts in the following order:

1. Triplet model embeddings

```
python src/compute_embbeding.py --config_file=config/08_efficientnetbo_triplet_pretrained_test.yaml
python src/compute_embbeding.py --config_file=config/08_efficientnetbo_triplet_pretrained_test.yaml
```

2. Autoencoder model embeddings

```
python src/compute_embbeding.py --config_file=config/10_efficientnetbo_autoenco_pretrained_train.yaml
python src/compute_embbeding.py --config_file=config/11_efficientnetbo_autoenco_pretrained_test.yaml
```

3. SwaV model embeddings

```
python src/compute_SwaV.py
```

4. Semantic attributes from binary model

```
python src/compute_semantic_attributes.py
```

Once we have computed the embedding from all model and also the semantic attributes, now we will concat the embeddings and semantic attributes. 
```
python src/concat_embeddings.py
```

Find bellow the list of all embeddings computed in the scripts:


> Triplet embeddings

> Autoencoder embeddings

> SwaV embeddings

> Semantic attributes

> Triplet embeddings + Semantic attributes

> Autoencoder embeddings + Semantic attributes

> SwaV embeddings + Semantic attributes


#### In order to performe some sanity checks on the embeddings, run the the command:
```
python .\src\test_embeddings.py
```
This command will compare if the target classes are correct across the train and test set.

### Validation
```
python .\src\evaluate_embeddings.py
```

# DVC - Data Versioning Controll

In order to pull the data used and generated on this repo, and also manage the data versioning in your experiments use dvc cli.

First you need to install dvc. Use the this link: [DVC Install](https://dvc.org/doc/install).

Once you have DVC installed, you should navigate to the root of this directory and type and click enter on the command bellow to set up the remote repositore where the data is stored. We are using Google Drive for this project.


```
dvc remote add --default myremote  gdrive://0AIac4JZqHhKmUk9PDA
dvc remote modify myremote gdrive_acknowledge_abuse true
```

To pull the data use:
```
dvc pull
```

To track a new data or change on the current data you should use the command bellow:
```
dvc add new_data
```

To keep things simple you can use the command `dvc-add` to loop over all data folders and files and add them to the version control.
````
dvc-add.bat # Windows
dvc-add.sh # Linux
````
