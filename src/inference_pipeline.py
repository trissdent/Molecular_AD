# Inference pipeline.
# Modify for your task.

import torch

from configs import ConfigReader
from shared.services.data import Transformer
from shared.services.models_hub import UNet


def load_model(config, device):
    model = UNet(
        n_channels=config.model.n_channels,
        n_classes=config.model.n_classes
    )
    model.load(config.training.checkpoint_dir + "model_weights.pt")
    model.to(device)
    model.eval()
    return model


def predict(model, data, transform, device):
    x = transform(data)
    x = x.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    return output


def run(config_path="./configs/defaults.yml", experiment_path=None):
    # Load config
    config = ConfigReader.merge(config_path, experiment_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transform
    transform = Transformer(
        target_size=tuple(config.transform.target_size),
        do_augmentation=False
    )
    
    # Load model
    model = load_model(config, device)
    
    # Load your data here
    # data = Image.open("image.jpg")      # Image
    # data = load_audio("audio.wav")      # Audio
    # data = load_text("text.txt")        # Text
    # data = np.load("data.npy")          # Numpy
    data = None  # Modify this
    
    # Predict
    output = predict(model, data, transform, device)
    
    # Process output (modify for your task)
    # Classification: pred = output.argmax(dim=1)
    # Segmentation: pred = torch.sigmoid(output) > 0.5
    print(output)


if __name__ == "__main__":
    run(config_path="./configs/defaults.yml")