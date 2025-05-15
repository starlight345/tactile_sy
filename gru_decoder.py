import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import wandb
import os
import pandas as pd
import numpy as np
from tabnet_models import TabNet  # Assume this is the existing TabNet inverse implementation
from torch.nn.utils.rnn import pad_sequence
import glob

class TabNetGRUSmilesDecoder(nn.Module):
    def __init__(self, tabnet_encoder, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.1):
        super(TabNetGRUSmilesDecoder, self).__init__()

        self.tabnet_encoder = tabnet_encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc_latent_to_hidden = nn.Linear(self.tabnet_encoder.final_out_dim, hidden_dim * num_layers)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, properties, input_token_ids):
        B, T = input_token_ids.shape

        with torch.no_grad():
            latent_z, _ = self.tabnet_encoder(properties)

        embedded = self.embedding(input_token_ids)  # (B, T, E)
        h0 = self.fc_latent_to_hidden(latent_z)  # (B, H*num_layers)
        # h0 = h0.view(B, -1, self.gru.num_layers).permute(2, 0, 1).contiguous()  # (num_layers, B, H)
        h0 = h0.view(B, self.gru.num_layers, -1).permute(1, 0, 2).contiguous()

        outputs, _ = self.gru(embedded, h0)
        logits = self.output_layer(outputs)  # (B, T, vocab_size)

        return logits


class PropertySmilesDataset(Dataset):
    def __init__(self, df):
        self.properties = torch.tensor(np.stack(df['properties'].values)).float()
        # self.smiles_tokens = torch.tensor(np.stack(df['smiles_tokens'].values)).long()
        self.smiles_tokens = [torch.tensor(x).long() for x in df['smiles_tokens'].values]

    def __len__(self):
        return len(self.properties)

    def __getitem__(self, idx):
        return self.properties[idx], self.smiles_tokens[idx]
def debug_one_batch(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch_idx, (properties, smiles_input) in enumerate(dataloader):
        print(f"[DEBUG] Got batch {batch_idx}")
        print(f"properties.shape = {properties.shape}")
        print(f"smiles_input.shape = {smiles_input.shape}")

        properties = properties.to(device)
        smiles_input = smiles_input.to(device)

        target = smiles_input[:, 1:]
        input_seq = smiles_input[:, :-1]

        optimizer.zero_grad()
        logits = model(properties, input_seq)

        print(f"logits.shape = {logits.shape}")
        print(f"logits.mean = {logits.mean().item():.4f}, std = {logits.std().item():.4f}")

        loss = criterion(logits.view(-1, logits.size(-1)), target.contiguous().view(-1))
        print(f"loss = {loss.item():.4f}")

        loss.backward()
        optimizer.step()
        print("[DEBUG] One batch finished successfully ✅")
        break  # 한 배치만!

# def train_one_epoch(model, dataloader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
#     for properties, smiles_input in dataloader:
#         properties = properties.to(device)
#         smiles_input = smiles_input.to(device)
#         target = smiles_input[:, 1:]
#         input_seq = smiles_input[:, :-1]

#         optimizer.zero_grad()
#         logits = model(properties, input_seq)
#         loss = criterion(logits.view(-1, logits.size(-1)), target.contiguous().view(-1))
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#     return total_loss / len(dataloader)
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch=0, use_wandb=False):
    model.train()
    total_loss = 0

    for batch_idx, (properties, smiles_input) in enumerate(dataloader):
        properties = properties.to(device)
        smiles_input = smiles_input.to(device)
        target = smiles_input[:, 1:]
        input_seq = smiles_input[:, :-1]

        optimizer.zero_grad()
        logits = model(properties, input_seq)
        loss = criterion(logits.view(-1, logits.size(-1)), target.contiguous().view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # === wandb 로깅 추가 ===
        if use_wandb:
            wandb.log({"batch_train_loss": loss.item()}, step=epoch * len(dataloader) + batch_idx)

        # === 중간 출력 ===
        if batch_idx % 10 == 0:
            print(f"[Epoch {epoch}] Batch {batch_idx} Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)



def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for properties, smiles_input in dataloader:
            properties = properties.to(device)
            smiles_input = smiles_input.to(device)
            target = smiles_input[:, 1:]
            input_seq = smiles_input[:, :-1]

            logits = model(properties, input_seq)

            print("Logits shape:", logits.shape)
            print("Logits sample:", logits[0, :5])  # 샘플 일부 출력
            loss = criterion(logits.view(-1, logits.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()

            ## For debugging
            print("logits mean:", logits.mean().item(), "std:", logits.std().item())
            # break  # 일단 1배치만 보기
    return total_loss / len(dataloader)

def collate_fn(batch):
    props, smiles = zip(*batch)
    props = torch.stack(props)
    smiles = pad_sequence(smiles, batch_first=True, padding_value=0)  # PAD_ID = 0
    return props, smiles



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--property_dim', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--exp_name', type=str, default='gru-decoder')
    parser.add_argument('--data_path', type=str, default='../tactile/polyOne_tokenized')
    parser.add_argument('--num_files', type=int, default=2)
    args = parser.parse_args()

    print("Start training")

    if args.wandb:
        print("Initializing wandb...")
        wandb.init(project="TabNet_SmilesDecoder", name=args.exp_name)
        wandb.config.update(vars(args))

    print("Loading TabNet model...")
    tabnet = TabNet(inp_dim=args.property_dim, final_out_dim=600).to(args.device)
    tabnet.eval()
    for p in tabnet.parameters():
        p.requires_grad = False
    print("Initializing GRU decoder...")
    model = TabNetGRUSmilesDecoder(tabnet, vocab_size=args.vocab_size).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # df = pd.read_parquet(args.data_path)
    print("Loading dataset files...")
    all_files = sorted(glob.glob(os.path.join(args.data_path, '*.parquet')))
    selected_files = all_files[:args.num_files]
    print("Reading and concatenating parquet files...")
    df = pd.concat([pd.read_parquet(f) for f in selected_files])
    dataset = PropertySmilesDataset(df)
    print("Finished loading data. Total samples:", len(df))

    print("Creating dataset and dataloaders...")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    # print("Running a debug one-batch training...")
    # debug_one_batch(model, train_loader, optimizer, criterion, args.device)


    print("Starting training loop...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device, epoch=epoch, use_wandb=args.wandb)
        val_loss = validate(model, val_loader, criterion, args.device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if args.wandb:
            wandb.log({"epoch_train_loss": train_loss, "val_loss": val_loss}, step=epoch)

    print("Training complete.")


if __name__ == "__main__":
    main()
