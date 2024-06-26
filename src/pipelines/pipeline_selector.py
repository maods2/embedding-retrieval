from pipelines.triplet import train_triplet
from pipelines.auto_encoder import train_auto_encoder
from pipelines.swav import train_swav

def get_pipeline(argument):
    
    if argument == "triplet":
        return train_triplet
    
    if argument == "autoencoder":
        return train_auto_encoder
    
    if argument == "swav":
        return train_swav

    else:
        raise Exception("Not found")