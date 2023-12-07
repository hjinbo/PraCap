import torch
import clip
import math
import json
import pickle
import pandas as pd
from tqdm import tqdm

def load_captions(path):   
    with open(path, 'r') as infile:
        data = json.load(infile)
        annotations = data['annotations']
    punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', ' ', '-', '.', '/', ':', 
                    ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    captions = []
    for caption_info in annotations:                  
        caption = caption_info['caption']
        caption = caption.strip()
        if caption.isupper():
            caption = caption.lower()
        caption = caption[0].upper() + caption[1:]
        if caption[-1] not in punctuations:
            caption += '.'
        captions.append(caption)
    return captions

def compose_prompts(tokenizer, prompt_captions):
    prompt_head = 'Similar images show'
    prompt_tail = ' This image shows'
    if len(prompt_captions) == 0:
        prompt = prompt_head + ' something.' + prompt_tail
    else:
        prompt = ''
        for p_caption in prompt_captions:
            prompt += ' ' + p_caption[:-1] + ','
        prompt = prompt[:-1] + '.'
        prompt = prompt_head + prompt + prompt_tail

    prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.int64)
    return prompt_tokens

# find k most similar vector from support_memory via features and then weighted sum
def projection(feature, support_memory, k=10):
    if k == 0:
        sim = feature @ support_memory.T.float()
        sim = (sim * 100).softmax(dim=-1)
        embedding = sim @ support_memory.float()
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding
    most_sim = []
    sim = feature @ support_memory.T.float()
    for _ in range(k):
        _, max_id = torch.max(sim, dim=1)
        most_sim.append(support_memory[max_id.item()].unsqueeze(0))
        sim[0][max_id.item()] = 0
    most_sim = torch.cat(most_sim) # (k, 512)
    new_sim = feature @ most_sim.T.float()
    new_sim = (new_sim * 100).softmax(dim=-1)
    embedding = new_sim @ most_sim.float()
    embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding

def get_support_memory(mode='train', clip_model_type='ViT-B/32', dataset='coco'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False, download_root='./pretrained/clip/')
    try:
        with open(f'others/{dataset}_support_memory.pkl', 'rb') as f:
            support_memory = pickle.load(f)
            support_memory = support_memory['support_memory'].float()
    except:
        ## construct the support memory
        with open(f'data/{dataset}/outfiles/captions_{dataset}_{mode}.json', 'r') as f:
            data = json.load(f)
        annotations = data['annotations']
        data = [ann['caption'] if len(ann['caption']) < 77 else ann['caption'][:100] for ann in annotations]
        # data = random.sample(data, 80000)
        text_features = []
        batch_size = 1000
        clip_model.eval()
        for i in tqdm(range(0, len(data[:]) // batch_size)):
            texts = data[i * batch_size:(i + 1) * batch_size]
            with torch.no_grad():
                texts_token = clip.tokenize(texts).to(device)
                text_feature = clip_model.encode_text(texts_token)
                text_features.append(text_feature)
        support_memory = torch.cat(text_features, dim=0)
        support_memory /= support_memory.norm(dim=-1, keepdim=True).float()
        with open(f'others/{dataset}_support_memory.pkl', 'wb') as f:
            pickle.dump({'support_memory': support_memory}, f)
    return support_memory

def get_image_sim_texts(feature, captions, support_memory, k=5, threshold=0.0):
    most_sim_texts = []
    sim = feature @ support_memory.T.float()
    for _ in range(k):
        max_sim, max_id = torch.max(sim, dim=1)
        # if max_sim.item() < threshold:
        #     continue
        most_sim_texts.append(captions[max_id.item()])
        sim[0][max_id.item()] = 0
    return most_sim_texts[:-1]


if __name__ == '__main__':
    get_support_memory(dataset='minicoco')