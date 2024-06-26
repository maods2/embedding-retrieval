from pathlib import Path
import torch
import time
import matplotlib.pyplot as plt
import yaml

def slice_image_paths(paths):
    return [i.split('/')[-1].replace('\\','/') for i in paths]


def save_checkpoint(model, optimizer, loss, config):
    timestamp_folder = create_timestamp_folder(config)
    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer is not None else None}
    Path(f'./checkpoints/embedding_models/{timestamp_folder}').mkdir(parents=True, exist_ok=True)
    filename=f"./checkpoints/embedding_models/{timestamp_folder}/checkpoint.pth.tar"
    draw_loss_curve(
        history=loss, 
        results_path=f"./checkpoints/embedding_models/{timestamp_folder}"
        )
    torch.save(state, filename)

    with open(f"./checkpoints/embedding_models/{timestamp_folder}/config.yaml", "w") as yaml_file:
        yaml.dump(config.__dict__, yaml_file)




def create_timestamp_folder(config):
    """
    Create a folder name based on the current timestamp.
    Returns:
        folder_name (str): The name of the folder, in the format 'YYYY-MM-DD-HH-MM-SS'.
    """
    current_time = time.localtime()
    folder_name = time.strftime('%Y-%m-%d-%H-%M-%S', current_time)
    return f'{config.model}_{config.pipeline}_{folder_name}'    

def draw_loss_curve(history,results_path):

    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig(results_path + '/loss.png')
