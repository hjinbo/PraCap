import json
import os
import clip
from PIL import Image
import pickle
import torch
from tqdm import tqdm

@torch.no_grad()
def images_features_extract(rootpath, device, encoder, proprecess, annotations, outpath):
    results = []
    progress = tqdm(total=len(annotations))
    for image_id in annotations:
        captions = annotations[image_id]
        image_path = rootpath + image_id
        image = Image.open(image_path)
        image_for_clip = proprecess(image).unsqueeze(dim = 0).to(device)
        image_features = encoder.encode_image(image_for_clip).squeeze(dim = 0).to('cpu')
        image_features /= image_features.norm(2, -1)
        results.append([image_id, image_features, captions])
        progress.update()
    progress.close()

    with open(outpath, 'wb') as outfile:
        pickle.dump(results, outfile)

def main(mode, dataset_name):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    clip_type = 'ViT-B/32'

    assert mode in ['train', 'val', 'test'], 'invalid mode input! (opt: train, val, test)'
    assert dataset_name in ['minicoco', 'coco', 'flickr30k'], 'invalid dataset name input! (opt: minicoco, coco, flickr30k)'
    in_path = f'data/{dataset_name}/outfiles/captions_{dataset_name}_{mode}_format.json'
    out_path = f'XXCap/preprocess_outputs/{dataset_name}/{dataset_name}_image_features.pkl'
    image_root_path = f'data/{dataset_name}/images/'
    with open(in_path, 'r') as infile:
        images = json.load(infile)
    
    if os.path.exists(out_path):
        with open(out_path, 'rb') as infile:
            images_features = pickle.load(infile)
        print(f'Already extract image features, the length: {len(images_features)}')
    else:
        encoder, proprecess = clip.load(clip_type, device, download_root='pretrained/clip')
        images_features_extract(image_root_path, device, encoder, proprecess, images, out_path)

def preprocess(mode, dataset_name):
    assert mode in ['train', 'val', 'test'], 'invalid mode input! (opt: train, val, test)'
    assert dataset_name in ['minicoco', 'coco', 'flickr30k'], 'invalid dataset name input! (opt: minicoco, coco, flickr30k)'
    preprocess_path = f'data/{dataset_name}/outfiles/captions_{dataset_name}_{mode}.json'
    preprocess_outpath = f'data/{dataset_name}/outfiles/captions_{dataset_name}_{mode}_format.json'
    with open(preprocess_path, 'r') as infile:
        data = json.load(infile)
        annotations = data['annotations']
    image_captions_map = dict()
    for caption_info in annotations:
        image_id = caption_info['image_id']
        temp_list = image_captions_map[image_id] if image_id in image_captions_map.keys() else []
        temp_list.append(caption_info['caption'])
        image_captions_map.update({image_id: temp_list})
    with open(preprocess_outpath, 'w') as outfile:
        json.dump(image_captions_map, outfile)



if __name__ == '__main__':
    preprocess(mode='test', dataset_name='minicoco')
    main(mode='test', dataset_name='minicoco')
    