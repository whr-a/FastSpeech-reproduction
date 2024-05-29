import os
import soundfile
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_xlarge")

def transcribe_audio_files(audio_dir, output_file):

    with open(output_file, "w", encoding="utf-8") as f:

        for i in range(100):
            audio_path = os.path.join(audio_dir, f"0_{i}_waveglow.wav")
            print("Transcribing:", audio_path)
            text, *_ = asr_model.transcribe([audio_path])[0]
            f.write(text + "\n")

    print("Transcription complete. Results saved to:", output_file)

audio_directory = "/home/whr-a/TTS/FastSpeech/results"

output_text_file = "/home/whr-a/TTS/STTresult/result.txt"

transcribe_audio_files(audio_directory, output_text_file)
# Average Word Error Rate (WER): 0.08063332113898879