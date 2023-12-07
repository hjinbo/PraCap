from torch.utils.data import Dataset, DataLoader
import pickle
import torch
from preprocess.utils import *
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


class CaptionDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.all_data = pickle.load(f)
        self.captions = [data[1] for data in self.all_data]
        self.clip_caption_features = [data[2] for data in self.all_data]
        self.tokenizer = _Tokenizer()
        self.bos = torch.tensor([self.tokenizer.encoder['<|startoftext|>']], dtype=torch.int64)
        self.eos = torch.tensor([self.tokenizer.encoder['<|endoftext|>']], dtype=torch.int64)
        # tokenize simtexts
        # self.simtexts = [data[0] for data in self.all_data]
        self.token_prompts = [torch.cat((self.bos, compose_prompts(self.tokenizer, data[0]))) for data in self.all_data]
        # tokenize caption
        self.token_captions = [torch.cat((torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64), 
                                          self.eos)) for caption in self.captions]
        # detect_entities + caption tokenized
        self.token_adds = []
        for i in range(len(self)):
            self.token_adds.append(torch.cat((self.token_prompts[i], self.token_captions[i])))
        
        all_token_len = torch.tensor([len(self.token_adds[i]) for i in range(len(self))]).float()
        self.max_token_len = min(int(all_token_len.mean() + all_token_len.std() * 10), int(all_token_len.max()))

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, item):
        clip_caption_feature = self.clip_caption_features[item].float()
        clip_caption_feature /= clip_caption_feature.norm(2, -1)

        token_adds, token_prompt = self.pad_tokens(item)
        return clip_caption_feature, token_adds, token_prompt
    
    def pad_tokens(self, item):
        token_adds = self.token_adds[item]
        token_prompt = self.token_prompts[item]

        padding = self.max_token_len - token_adds.shape[0]
        if padding > 0:
            token_adds = torch.cat((token_adds, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            token_adds = token_adds[:self.max_token_len]
            
        padding = self.max_token_len - token_prompt.shape[0]
        if padding > 0:
            token_prompt = torch.cat((token_prompt, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            token_prompt = token_prompt[:self.max_token_len]

        return token_adds, token_prompt
    

def get_data_loader(mode, dataset_name, batch_size=4, shuffle=True, num_workers=0):
    assert mode in ['train', 'val', 'test'], 'invalid mode input! (opt: train, val, test)'
    assert dataset_name in ['minicoco', 'coco', 'flickr30k'], 'invalid dataset name input! (opt: minicoco, coco, flickr30k)'
    data_path = f'XXCap/preprocess_outputs/{dataset_name}/{dataset_name}_text_features_with_prompts.pkl'
    dataset = CaptionDataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == '__main__':
    dataloader = get_data_loader('train', batch_size=2)
    it = iter(dataloader)
    clip_caption_feature, token_adds, token_prompt = next(it)
    print(token_adds, token_prompt)