import os
import soundfile
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_xlarge")


def transcribe_audio_files(audio_dir, output_file):
    # Load the pretrained STT model

    # Open the output file for writing
    with open(output_file, "w", encoding="utf-8") as f:
        # Iterate over specified range of numbers
        for i in range(100):  # assuming you want to process files from 0 to 99
            # Construct the audio file path
            audio_path = os.path.join(audio_dir, f"0_{i}_waveglow.wav")
            print("Transcribing:", audio_path)

            # Write the transcription result to the output file
            text, *_ = asr_model.transcribe([audio_path])[0]
            f.write(text + "\n")

    print("Transcription complete. Results saved to:", output_file)

# 设置音频文件所在文件夹路径
audio_directory = "/home/whr-a/TTS/FastSpeech/results"
# 设置输出的文本文件路径
output_text_file = "/home/whr-a/TTS/STTresult/result.txt"
# 调用函数进行语音转文本
transcribe_audio_files(audio_directory, output_text_file)
# Average Word Error Rate (WER): 0.08063332113898879