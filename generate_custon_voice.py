from bark.api import generate_audio
from transformers import BertTokenizer
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic

# Enter your prompt and speaker here
text_prompt = "N世界是一款元宇宙产品，提供线上3D展馆，为企业和个人提供3D展示和交互式体验。通过打造更具沉浸式的元宇宙场景，为用户带来更真实、直观的体验感，提供更丰富、更创新的展示方式。"
voice_name = "zh_speaker_10" # use your custom voice name here if you have one

# load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# download and load all models
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

# simple generation
audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)

# generation with more control
x_semantic = generate_text_semantic(
    text_prompt,
    history_prompt=voice_name,
    temp=0.7,
    top_k=50,
    top_p=0.95,
)

x_coarse_gen = generate_coarse(
    x_semantic,
    history_prompt=voice_name,
    temp=0.7,
    top_k=50,
    top_p=0.95,
)
x_fine_gen = generate_fine(
    x_coarse_gen,
    history_prompt=voice_name,
    temp=0.5,
)
audio_array = codec_decode(x_fine_gen)

from IPython.display import Audio
# play audio
Audio(audio_array, rate=SAMPLE_RATE)

from scipy.io.wavfile import write as write_wav
# save audio
filepath = "/output/audio.wav" # change this to your desired output path
write_wav(filepath, SAMPLE_RATE, audio_array)

