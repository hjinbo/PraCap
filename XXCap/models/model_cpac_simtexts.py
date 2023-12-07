import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional, List
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class CaptionModel(nn.Module):
    def __init__(self, dictionary):
        super(CaptionModel, self).__init__()
        self.embed_size = dictionary['embed_size']
        self.vocab_size = dictionary['vocab_size']
        self.clip_project = MLP((512, self.embed_size))
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.positional_encoding = PositionalEncoding(d_model=self.embed_size, dropout=0)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_size, nhead=4, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=1)
        self.dropout = nn.Dropout(dictionary['dropout'])
        self.decoder_linear = nn.Linear(self.embed_size, self.vocab_size)

    def forward(self, clip_embeddings, tokens):
        tgt_key_padding_mask = self.get_key_padding_mask(tokens).to(device)
        tgt_mask = self.generate_square_subsequent_mask(tokens.size(1)).to(device)
        embeddings = self.positional_encoding(self.embedding(tokens)).to(device)
        clip_embeddings = self.clip_project(clip_embeddings).unsqueeze(1).to(device)
        out = self.transformer_decoder(memory=clip_embeddings, tgt=embeddings, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        out = self.decoder_linear(self.dropout(out))
        return out
    
    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters]) / 1_000_000
        print('Trainable parameters: %.3fM' % params)

    def get_key_padding_mask(self, tokens):
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 0] = -torch.inf
        return key_padding_mask

    def generate_square_subsequent_mask(self, token_size):
        return nn.Transformer.generate_square_subsequent_mask(token_size)


    def inference(self, image_features, prompt_tokens, tokenizer=None):
        if tokenizer is None:
            return self.greedy_search(image_features, prompt_tokens)
        beam_search_results = self.beam_search(image_features, prompt_tokens, tokenizer)
        return beam_search_results

    # batch_size = 1
    def greedy_search(self, image_features, prompt_tokens, max_length=50):
        image_features = self.clip_project(image_features.unsqueeze(1)) # (b, 1, 512)
        tokens = prompt_tokens
        for i in range(max_length):
            decoder_input = self.positional_encoding(self.embedding(tokens))
            out = self.transformer_decoder(memory=image_features, tgt=decoder_input)
            out = self.decoder_linear(self.dropout(out[:, -1]))
            prediction = torch.argmax(out, dim=1, keepdim=True) # (b, 1)
            tokens = torch.cat((tokens, prediction), dim=1)
            if prediction[0] == 49407:
                break
        # 截取prompts后的部分
        tokens = tokens[:, len(prompt_tokens[0]):]
        return tokens
    
    def beam_search(self, image_features, prompt_tokens, tokenizer, temperature=1.0, max_len=50, beam_width=5, end_of_sentences=['.', ' .']):
        eos = [tokenizer.encode(end_of_sentence)[-1] for end_of_sentence in end_of_sentences]
        scores = None
        seq_lengths = torch.ones(beam_width, device=device)
        is_stopped = torch.zeros(beam_width, device=device, dtype=torch.bool)
        tokens = prompt_tokens
        image_features = self.clip_project(image_features.unsqueeze(1)).to(device)
        
        generated = self.positional_encoding(self.embedding(tokens)).to(device)
        for i in range(max_len):
            outputs = self.transformer_decoder(memory=image_features, tgt=generated)
            outputs = self.decoder_linear(self.dropout(outputs[:, -1]))
            outputs = outputs / (temperature if temperature > 0 else 1.0)
            logits = outputs.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_width, -1)
                image_features = image_features.expand(beam_width, *image_features.shape[1:])
                generated = generated.expand(beam_width, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)

                tokens = tokens.expand(beam_width, *tokens.shape[1:])
                tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_width, -1)
                next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='trunc')
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = self.positional_encoding(self.embedding(next_tokens)).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            assert len(eos) == 2 # hack
            is_stopped = is_stopped + (next_tokens.eq(eos[0]) | next_tokens.eq(eos[1])).squeeze()
            if is_stopped.all():
                break
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [output[:int(length)] for output, length in zip(output_list, seq_lengths)]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]
        # print([output_texts[0]])
        return [output_texts[0][len(prompt_tokens[0]):]]