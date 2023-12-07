import torch
from torch.nn import functional as nnf
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.optim import AdamW
import os
import argparse
import sys
from models.model_cpac_simtexts import CaptionModel
from datasets.dataset_cpac_simtexts import get_data_loader
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(model, args, warmup_steps=5000):
    model.summary()
    epochs = args.epochs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    train_dataloader = get_data_loader(mode='train', dataset_name=args.dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch + 1}")
        sys.stdout.flush()
        total_loss = 0
        progress = tqdm(total=len(train_dataloader), desc='')
        for idx, (clip_caption_feature, token_adds, _) in enumerate(train_dataloader):
            clip_caption_feature = clip_caption_feature.to(device)
            token_adds = token_adds.to(device).long()
            outputs = model(clip_caption_feature, token_adds[:, :-1])
            outputs = outputs.reshape(-1, outputs.shape[-1])
            token_adds = token_adds[:, 1:].flatten()
            loss = nnf.cross_entropy(outputs, token_adds)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            total_loss += loss.item()
        progress.close()
        avg_loss = total_loss / len(train_dataloader)
        if (epoch + 1) % args.save_every == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.output_prefix}-{avg_loss:.2f}.pt"))
    return model



def main():
    dataset = 'minicoco'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default=f'XXCap/checkpoints/{dataset}/cpac_simtexts/')
    parser.add_argument('--output_prefix', default='transformer_decoder', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dataset', default=dataset)

    parser.add_argument('--embed_size', default=512, type=int, help='dimension for word embedding vector')
    parser.add_argument('--dropout', default=0.1, type=float, help='')
    args = parser.parse_args()
    _tokenizer = _Tokenizer()
    args_dict = vars(args)
    args_dict['vocab_size'] = len(_tokenizer.encoder)
    model = CaptionModel(args_dict)
    # model.load_state_dict(torch.load(f'XXCap/checkpoints/{dataset}/cpac_simtexts/transformer_decoder-1.15.pt', map_location='cpu'))
    train(model, args)
    

if __name__ == '__main__':
    main()