python ./src/main.py --pipeline=train --config-file=./config/train_autoencoder_efficientnet.toml
python ./src/main.py --pipeline=train --config-file=./config/train_swav_efficientnet.toml
python ./src/main.py --pipeline=train --config-file=./config/train_triplet_efficientnet.toml

python ./src/main.py --pipeline=compute_semantic_attributes --config-file=./config/inference_semantic_attributes_train_data.toml
python ./src/main.py --pipeline=compute_semantic_attributes --config-file=./config/inference_semantic_attributes_test_data.toml

python ./src/main.py --pipeline=concat --config-file=./config/concat_embeddings.toml



