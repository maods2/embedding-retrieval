@echo off
REM List of folders to add to DVC
set FOLDERS=.\checkpoints\embedding_models\ ^
.\checkpoints\semantic_models\ ^
.\data\terumo-data-testset\ ^
.\data\terumo-data-training\ ^
.\embeddings\efficientnetb0_4096_autoencoder_test.pickle ^
.\embeddings\efficientnetb0_4096_autoencoder_train.pickle ^
.\embeddings\efficientnetb0_4096_pretrained_test.pickle ^
.\embeddings\efficientnetb0_4096_pretrained_train.pickle ^
.\embeddings\efficientnet_SwaV_test.pickle ^
.\embeddings\efficientnet_SwaV_train.pickle ^
.\embeddings\semantic_att_efficientnetb0_encoder_test.pickle ^
.\embeddings\semantic_att_efficientnetb0_encoder_train.pickle ^
.\embeddings\semantic_att_efficientnetb0_test.pickle ^
.\embeddings\semantic_att_efficientnetb0_train.pickle ^
.\embeddings\semantic_att_efficientnet_SwaV_test.pickle ^
.\embeddings\semantic_att_efficientnet_SwaV_train.pickle ^
.\embeddings\semantic_test.pickle ^
.\embeddings\semantic_train.pickle 

REM Loop through each folder and add to DVC
for %%f in (%FOLDERS%) do (
    dvc add %%f
)


