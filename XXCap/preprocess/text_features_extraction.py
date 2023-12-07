import os
import pickle
import clip
import torch
from tqdm import tqdm
from utils import get_support_memory, load_captions

# find k texts that most similar with each text via clip similarity
@torch.no_grad()
def topk_texts_extract(dataset, captions, out_path, k=5):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    encoder, _ = clip.load('ViT-B/32', device=device, download_root='pretrained/clip')
    support_memory = get_support_memory(mode='train', dataset=dataset)
    new_captions = []
    progress = tqdm(total=len(captions))
    for caption in captions:
        token = clip.tokenize(caption, truncate=True).to(device)
        feature = encoder.encode_text(token).float() # (1, 512)
        most_sim_texts = []
        sim = feature @ support_memory.T.float()
        for _ in range(k):
            _, max_id = torch.max(sim, dim=1)
            most_sim_texts.append(captions[max_id.item()])
            sim[0][max_id.item()] = 0
        new_captions.append([most_sim_texts[1:], caption, feature.squeeze(0).cpu()]) # exclude the same text
        progress.update()
    progress.close()
    with open(out_path, 'wb') as outfile:
        pickle.dump(new_captions, outfile)


def main(mode, dataset_name):
    assert mode in ['train', 'val', 'test'], 'invalid mode input! (opt: train, val, test)'
    assert dataset_name in ['minicoco', 'coco', 'flickr30k'], 'invalid dataset name input! (opt: minicoco, coco, flickr30k)'
    captions_path = f'data/{dataset_name}/outfiles/captions_{dataset_name}_{mode}.json'
    out_path = f'XXCap/preprocess_outputs/{dataset_name}/{dataset_name}_text_features_with_prompts.pkl'
    if os.path.exists(out_path):
        with open(out_path, 'rb') as infile:
            captions_with_simtexts = pickle.load(infile)
        print(captions_with_simtexts[:5])
        print(f'Already extract simtexts, the length of {mode} datasets: {len(captions_with_simtexts)}')
    else:
        captions = load_captions(captions_path)
        topk_texts_extract(dataset=dataset_name, captions=captions, out_path=out_path)

if __name__ == '__main__':
    main(mode='train', dataset_name='minicoco')