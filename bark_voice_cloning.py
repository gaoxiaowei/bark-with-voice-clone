from datetime import datetime

import modal
import nltk
from modal import Stub, Image
from scipy.io.wavfile import write as write_wav

from bark.generation import (
    preload_models, load_codec_model, SAMPLE_RATE
)
import numpy as np_local
from hubert.customtokenizer import CustomTokenizer
from hubert.hubert_manager import HuBERTManager
from hubert.pre_kmeans_hubert import CustomHubert

hubert_manager = HuBERTManager()


def install_dependencies():
    preload_models(
        text_use_gpu=True,
        text_use_small=False,
        coarse_use_gpu=True,
        coarse_use_small=False,
        fine_use_gpu=True,
        fine_use_small=False,
        codec_use_gpu=True,
        force_reload=False,
        path="models"
    )
    nltk.download('punkt')
    hubert_manager.make_sure_hubert_installed()
    hubert_manager.make_sure_tokenizer_installed()


bark_image = Image \
    .debian_slim() \
    .apt_install("git") \
    .pip_install("git+https://github.com/suno-ai/bark.git") \
    .pip_install("nltk") \
    .pip_install("fairseq") \
    .pip_install("audiolm-pytorch") \
    .run_function(install_dependencies)

stub = Stub("bark-runner", image=bark_image)

if stub.is_inside():
    from bark import generate_audio
    from bark.generation import SAMPLE_RATE, generate_fine, generate_coarse, generate_text_semantic, codec_decode
    import numpy as np
    import torchaudio
    import torch
    import modal
    from encodec.utils import convert_audio

device = 'cuda'


@stub.function(gpu="a10g", mounts=[
    modal.Mount.from_local_dir(
        "/Users/arshankhanifar/bark-with-voice-clone/voices",
        remote_path="/root/voices"
    )
])
def clone_voice():
    model = load_codec_model(use_gpu=True)

    # Load the HuBERT model
    hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt').to(device)

    # Load the CustomTokenizer model
    tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth').to(
        device)  # Automatically uses the right layers

    audio_filepath = f'/root/voices/{VOICE_NAME}.wav'  # the audio you want to clone (under 13 seconds)
    wav, sr = torchaudio.load(audio_filepath)

    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.to(device)

    semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)
    # %%
    # Extract discrete codes from EnCodec
    print("Extract discrete codes from EnCodec")
    with torch.no_grad():
        encoded_frames = model.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]
    # %%
    # move codes to cpu
    print("move codes to cpu")
    codes = codes.cpu().numpy()
    # move semantic tokens to cpu
    print("move semantic tokens to cpu")
    semantic_tokens = semantic_tokens.cpu().numpy()
    # %%

    output_path = f"/root/voices/{VOICE_NAME}.npz"
    print(f"saving the voice to {output_path}")
    np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)

    # Enter your prompt and speaker here

    print("using the voice")
    audio_array = generate_audio(text_prompt, history_prompt=output_path, text_temp=0.7, waveform_temp=0.7)
    return audio_array, codes, semantic_tokens


@stub.function(gpu="a10g", mounts=[
    modal.Mount.from_local_dir(
        "/Users/arshankhanifar/bark-with-voice-clone/voices",
        remote_path="/root/voices"
    )
])
def talk_fine():
    output_path = f"/root/voices/{VOICE_NAME}.npz"
    first_temp, temp, top_k = 0.7, 0.4, 65
    x_semantic = generate_text_semantic(
        text_prompt,
        history_prompt=output_path,
        temp=first_temp,
        top_k=top_k,
        top_p=0.95,
    )

    x_coarse_gen = generate_coarse(
        x_semantic,
        history_prompt=output_path,
        temp=first_temp,
        top_k=top_k,
        top_p=0.95,
    )
    x_fine_gen = generate_fine(
        x_coarse_gen,
        history_prompt=output_path,
        temp=temp,
    )
    return codec_decode(x_fine_gen)


@stub.function(gpu="a100", mounts=[
    modal.Mount.from_local_dir(
        "/Users/arshankhanifar/bark-with-voice-clone/voices",
        remote_path="/root/voices"
    )
])
def talk():
    print("using the voice")
    output_path = f"/root/voices/{VOICE_NAME}.npz"
    audio_array = generate_audio(text_prompt, history_prompt=output_path)
    return audio_array


CLONE, VOICE_NAME = False, "joerogan"

text_prompt = "The Washington Post is set to publish a story tomorrow taking aim at conservative " \
              "groups requesting documents from universities over their ties to “disinformation” " \
              "tracking and the US government, according to a source familiar with the matter."


@stub.local_entrypoint()
def main():
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filepath = f"output/{VOICE_NAME}-{ts}.wav"  # change this to your desired output path
    if CLONE:
        audio_array, codes, semantic_tokens = clone_voice.call()
        np_local.savez(
            f"voices/{VOICE_NAME}.npz",
            fine_prompt=codes,
            coarse_prompt=codes[:2, :],
            semantic_prompt=semantic_tokens
        )
    else:
        audio_array = talk_fine.call()
    print(f"writing to: {filepath}")
    write_wav(filepath, SAMPLE_RATE, audio_array)
