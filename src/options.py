import argparse
import yaml

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()


    def initialize(self):      
        self.parser.add_argument('--config_file', type=str, default='./embeddings/config/00_default.yaml', help='# of test examples.')
    
    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt 


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
def load_parameters(filename):
    with open(filename, 'r') as file:
        params = yaml.safe_load(file)
    return params