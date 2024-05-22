import re

def calculate_wer(hypothesis, reference):
    # Split sentences into words, ignoring punctuation and converting to lowercase
    hypothesis_words = re.findall(r'\w+', hypothesis.lower())
    reference_words = re.findall(r'\w+', reference.lower())

    # Create sets of unique words
    hypothesis_set = set(hypothesis_words)
    reference_set = set(reference_words)

    # Calculate the number of incorrect words
    incorrect_words = len(hypothesis_set.symmetric_difference(reference_set))

    # Calculate the Word Error Rate
    wer = incorrect_words / max(len(reference_words), 1)
    return wer

def average_wer(hypothesis_file, reference_file):
    total_wer = 0.0
    num_sentences = 0

    with open(hypothesis_file, 'r', encoding='utf-8') as hyp_file, open(reference_file, 'r', encoding='utf-8') as ref_file:
        for hyp_sentence, ref_sentence in zip(hyp_file, ref_file):
            wer = calculate_wer(hyp_sentence.strip(), ref_sentence.strip())
            total_wer += wer
            num_sentences += 1

    average_wer = total_wer / num_sentences
    return average_wer

# 文件路径
hypothesis_file = "/home/whr-a/TTS/STTresult/result.txt"
reference_file = "/home/whr-a/TTS/data.txt"

# 计算并打印平均 WER
avg_wer = average_wer(hypothesis_file, reference_file)
print("Average Word Error Rate (WER):", avg_wer)
# Average Word Error Rate (WER): 0.808498312920166