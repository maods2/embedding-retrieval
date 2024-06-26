
from dynaconf import Dynaconf
from argparse import ArgumentParser
from pathlib import Path
from train.train import train_and_save_embeddings
from inference.compute_embbeding import compute_embeddings
from inference.compute_semantic_attributes import compute_semantic_attributes
from utils.concat_embeddings import concat_embeddings

def configParser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config-file", type=str, help="Config (TOML) file to read from.", required=True)
    parser.add_argument("--pipeline", type=str, help="Directory to save the file tree associated with this training instance")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda) to run the training")
    return parser


parser = configParser()
args = parser.parse_args()
config = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_file=args.config_file,
    root_path='.',
)


if args.pipeline == "train":
    train_and_save_embeddings(config)

elif args.pipeline == "compute_embedding":
    compute_embeddings(config, config.mode)

elif args.pipeline == "compute_semantic_attributes":
    compute_semantic_attributes(config, config.mode)
    
elif args.pipeline == "concat":
    concat_embeddings(config)