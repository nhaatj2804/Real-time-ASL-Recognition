import pandas as pd
import torch

DATA = pd.read_csv('seq2seq/train_data.tsv', sep='\t', header=None, names=['input', 'target'])
DATA = DATA[1:]

def tokenize(text):
    return text.lower().strip().split()

all_input_words = []
all_target_words = []
for idx,row in DATA.iterrows():
    all_input_words.extend(tokenize(row['input']))
    all_target_words.extend(tokenize(row['target'])) # Đảm bảo đây là row['target']
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'

input_vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + list(set(all_input_words))

target_vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + list(set(all_target_words))

input_word2idx = {word : idx for idx,word in enumerate(input_vocab)}
input_idx2word = {idx : word for idx,word in input_word2idx.items()} # Cách này đúng

target_word2idx = {word : idx for idx,word in enumerate(target_vocab)}
target_idx2word = {idx : word for idx,word in enumerate(target_vocab)} # Cách xây dựng trực tiếp target_idx2word từ target_vocab

def encode_input_for_eval(sentence):
    return [input_word2idx.get(word, input_word2idx['<PAD>']) for word in sentence.lower().split()]



def evaluate(encoder, decoder, sentence, input_word2idx = input_word2idx, target_word2idx = target_word2idx, target_idx2word = target_idx2word, encode_input_func = encode_input_for_eval, max_len=20, device='cpu'): # đổi tên encode_input để tránh nhầm lẫn
    encoder.eval()
    decoder.eval()

    # Sử dụng hàm encode_input_func được truyền vào
    tokens = encode_input_func(sentence) # Đổi tên ở đây nếu cần
    print(f"Input sentence: '{sentence}'")
    print(f"Tokens: {tokens}")
    print(f"Input vocab size for check: {len(input_word2idx)}")


    inputs = torch.tensor(tokens).unsqueeze(0).to(device)
    lengths = torch.tensor([len(tokens)]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = encoder(inputs, lengths)
        decoder_input = torch.tensor([target_word2idx['<SOS>']]).to(device)
        decoder_hidden = hidden

        decoded_words = []
 

        for i in range(max_len):
            output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            top1 = output.argmax(1).item()
   

            if top1 not in target_idx2word:
                print(f"----> CRITICAL: Index {top1} is NOT a key in target_idx2word (max index is {len(target_idx2word)-1})")
            
            if top1 == target_word2idx['<EOS>']:
                print("Predicted EOS token.")
                break

            word = target_idx2word.get(top1, '<UNK>') # Default to <UNK>
            if word == '<UNK>':
                print(f"----> WARNING: Predicted token is <UNK> for index {top1}")
            decoded_words.append(word)

            decoder_input = torch.tensor([top1]).to(device)
    
    return ' '.join(decoded_words)

