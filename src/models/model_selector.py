
# from models.vgg16 import get_vgg16
from models.efficient_netb0 import get_efficient_netb0, get_efficient_netb0_encoder
# from models.vit import get_vit
from models.auto_encoders import get_conv_encoder
from models.SwaV import get_swav

def get_model(argument):
    
    # if argument == "vgg16":
    #     return get_vgg16()

    if argument == "efficientnetb0":
        return get_efficient_netb0()
    
    elif argument == "efficientnetb0_encoder":
        return get_efficient_netb0_encoder()

    # elif argument == "vit":
    #     return get_vit()
    
    elif argument == "conv_encoder":
        return get_conv_encoder()
    
    elif argument == "swav":
        return get_swav()
    

    
        
    else:
        raise Exception("Not found")