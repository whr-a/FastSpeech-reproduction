import os
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

d = ModelDownloader()

def transcribe_audio_files(audio_dir, output_file):
    # Load the pretrained STT model
    speech2text = Speech2Text(
        **d.download_and_unpack(task="asr", corpus="wsj"),
        maxlenratio=0.0,
        minlenratio=0.0,
        beam_size=20,
        ctc_weight=0.3,
        lm_weight=0.5,
        penalty=0.0,
        nbest=1
    )

    # Open the output file for writing
    with open(output_file, "w", encoding="utf-8") as f:
        # Iterate over specified range of numbers
        for i in range(100):  # assuming you want to process files from 0 to 99
            # Construct the audio file path
            audio_path = os.path.join(audio_dir, f"0_{i}_waveglow.wav")
            print("Transcribing:", audio_path)

            # Read the audio file
            speech, rate = soundfile.read(audio_path)

            # Perform speech-to-text transcription
            nbests = speech2text(speech)

            # Write the transcription result to the output file
            text, *_ = nbests[0]
            f.write(text + "\n")

    print("Transcription complete. Results saved to:", output_file)

# 设置音频文件所在文件夹路径
audio_directory = "/home/whr-a/TTS/FastSpeech/results"
# 设置输出的文本文件路径
output_text_file = "/home/whr-a/TTS/STTresult/result.txt"
# 调用函数进行语音转文本
transcribe_audio_files(audio_directory, output_text_file)