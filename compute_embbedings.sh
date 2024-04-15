

python src/compute_embbeding.py --config_file=config/08_efficientnetbo_triplet_pretrained_test.yaml
python src/compute_embbeding.py --config_file=config/09_efficientnetbo_triplet_pretrained_train.yaml

python src/compute_embbeding.py --config_file=config/10_efficientnetbo_autoenco_pretrained_train.yaml
python src/compute_embbeding.py --config_file=config/11_efficientnetbo_autoenco_pretrained_test.yaml

python src/compute_SwaV.py

python src/compute_semantic_attributes.py

python src/concat_embeddings.py

python src/evaluate_embeddings.py