import os
from tqdm import tqdm

from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import make_chunks


def remove_silence(audio):
    chucks = split_on_silence(audio, min_silence_len=2000, silence_thresh=-45)
    output = sum(chucks)
    return output


def merge_similar_audio(dataset_path, files):
    outputs = []
    for audio in files:
        sound = AudioSegment.from_file(os.path.join(dataset_path, audio))
        outputs.append(sound)
    return sum(outputs)


def split_audio(audio):
    chunks = make_chunks(audio, 1000)
    return chunks


def save_audio(file_name, chunks, save_path):
    save_file = os.path.join(save_path, file_name)
    os.makedirs(save_file, exist_ok=True)
    for idx, chunk in enumerate(chunks):
        if len(chunk) == 1000:
            chunk.export(f'{save_file}/{file_name}_{idx + 1}.wav', format='wav')


def clean_audios(dataset_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    files = os.listdir(dataset_path)
    list_file = {}
    for file in files:
        if '_' in file:
            if file.split('_')[0] in list_file.keys():
                list_file[file.split('_')[0]].append(file)
            else:
                list_file[file.split('_')[0]] = [file]
        else:
            list_file[file.split('.')[0]] = file

    for name, files in tqdm(list_file.items()):
        if isinstance(files, list):
            audio = merge_similar_audio(dataset_path, files)
        else:
            audio = AudioSegment.from_file(os.path.join(dataset_path, files))
        audio = remove_silence(audio)
        audio = audio.set_sample_width(2)
        audio = audio.set_frame_rate(48000)
        audio = audio.set_channels(1)
        chunks = split_audio(audio)
        save_audio(name, chunks, save_path)


if __name__ == '__main__':
    clean_audios('dataset/', 'clean_dataset/')
