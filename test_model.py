import torch
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks

from model import AudioModel

classes = {
    0: 'Abdollah', 1: 'Alireza', 2: 'Azra',
    3: 'Benyamin', 4: 'Davood', 5: 'Javad',
    6: 'Khadijeh', 7: 'Kiana',
    8: 'Maryam', 9: 'Matin', 10: 'Melika',
    11: 'MohammadN', 12: 'MohammadP', 13: 'Mohammadali',
    14: 'Mona', 15: 'Morteza', 16: 'Nahid', 17: 'Nima',
    18: 'Omid', 19: 'Parisa', 20: 'Parsa',
    21: 'Sajedeh', 22: 'Sajjad', 23: 'Shima', 24: 'Tara',
    25: 'Zahra', 26: 'Zeynab'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioModel().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()


def preprocess_audio(chunk):
    samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
    samples = samples / np.max(np.abs(samples))
    samples = torch.tensor(samples).unsqueeze(0).unsqueeze(0)
    return samples.to(device)


def predict(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(48000).set_channels(1)
    chunks = make_chunks(audio, 1000)

    predictions = []
    for chunk in chunks:
        tensor_input = preprocess_audio(chunk)
        with torch.no_grad():
            output = model(tensor_input)
            predicted_class = torch.argmax(output, dim=1).item()
            predictions.append(predicted_class)
    print(predictions)
    most_common_class = max(set(predictions), key=predictions.count)
    return classes[most_common_class]


file_path = "dataset/Parsa_1.ogg"
predicted_name = predict(file_path)
print(f"Predicted Speaker: {predicted_name}")
