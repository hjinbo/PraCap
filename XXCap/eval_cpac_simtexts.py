from tqdm import tqdm
import torch
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import argparse
import json
import os
from models.model_cpac_simtexts import CaptionModel
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import pickle
from preprocess.utils import get_support_memory
from preprocess.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def coco_metric(input_sentences, path_anna):
    coco_set = COCO(path_anna)
    tmp_file = 'pred_set_tmp'
    if not os.path.exists('cache/'):
        os.makedirs('cache/')
    with open('cache/' + tmp_file + '.json', 'w') as f:
        json.dump(input_sentences, f)
    result = 'cache/' + tmp_file + '.json'
    cocoRes = coco_set.loadRes(result)
    cocoEval = COCOEvalCap(coco_set, cocoRes)
    cocoEval.evaluate()
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    return out

def eval(model:CaptionModel, tokenizer, args):
    source_dataset = args.source_dataset
    target_dataset = args.target_dataset
    with open(f'XXCap/preprocess_outputs/{target_dataset}/{target_dataset}_image_features.pkl', 'rb') as f:
        val_data_loader = pickle.load(f)
    support_memory = get_support_memory(mode='train', dataset=target_dataset)
    captions_path = f'data/{target_dataset}/outfiles/captions_{target_dataset}_train.json'
    captions = load_captions(captions_path)
    
    model.to(device)
    model.eval()
    image_set = []
    predictions = []
    bos = torch.tensor([tokenizer.encoder['<|startoftext|>']], dtype=torch.int64)
    progress = tqdm(total=len(val_data_loader), desc='')
    with torch.no_grad():
        for batch_id, (image_id, image_features, test_captions) in enumerate(val_data_loader):
            
            image_features = image_features.unsqueeze(0).to(device).float()

            embedding = projection(image_features, support_memory, k=100)
            
            prompt_captions = get_image_sim_texts(image_features, captions, support_memory)

            prompt_token = compose_prompts(tokenizer, prompt_captions)
            prompt_token = torch.cat((bos, prompt_token)).unsqueeze(0).to(device)

            inference_outputs = model.inference(embedding, prompt_token)
            batch_sentence_output = []
            for result in inference_outputs:
                result = result[result > 0].tolist()
                sentence_one = tokenizer.decode(result[:-1])
                batch_sentence_output.append(sentence_one.strip())
            for id, sentence in enumerate(batch_sentence_output):
                if image_id not in image_set:
                    image_set.append(image_id)
                    pred = {'image_id': image_id, 'caption': sentence}
                    predictions.append(pred)
            progress.update()
    progress.close()
    test_path = f'data/{target_dataset}/outfiles/captions_{target_dataset}_test.json'
    coco_stat = coco_metric(predictions, test_path)
    return coco_stat


def main(args):
    args_dict = vars(args)
    _tokenizer = _Tokenizer()
    args_dict = vars(args)
    args_dict['vocab_size'] = len(_tokenizer.encoder)
    model = CaptionModel(args_dict)
    model.load_state_dict(torch.load(f'XXCap/checkpoints/{args.source_dataset}/cpac_simtexts/transformer_decoder-0.65.pt', map_location='cpu'))
    model.to(device)
    model.eval()
    coco_stat = eval(model, _tokenizer, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_size', default=512, type=int, help='dimension for word embedding vector')
    parser.add_argument('--dropout', default=0.1, type=float, help='')
    parser.add_argument('--source_dataset', default='minicoco', type=str, help='')
    parser.add_argument('--target_dataset', default='minicoco', type=str, help='')
    args = parser.parse_args()
    main(args)