from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
import torch
import torch.nn as nn

from torch.amp import autocast, GradScaler

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from copy import deepcopy
from glob import glob

import numpy as np
import pandas as pd
import argparse
import os

from tabnet_utils import *

from sklearn.metrics import r2_score
import wandb
from datetime import datetime
import faiss

parser = argparse.ArgumentParser(
    description="Arguments for Fine-tuning to predict polymer properties"
)
parser.add_argument("--T", type=int, default=1000, help="timesteps for Unet model")
parser.add_argument("--droprate", type=float, default=0.1, help="dropout rate for model")
parser.add_argument("--dtype", default=torch.float32)
parser.add_argument("--workers", default=4, type=int)

parser.add_argument("--epochs", type=int, default=100, help="total epochs for training")
parser.add_argument("--start_epoch", type=int, default=0, help="start epochs for training")
parser.add_argument("--interval", type=int, default=5, help="epochs interval for evaluation")
parser.add_argument("--steps", type=int, default=1024, help="train steps per epoch")

parser.add_argument("--num_classes", type=int, default=37, help="number of classes for fine-tuning model")
parser.add_argument("--freeze", default=True, type=lambda s: s in ["True", "true", 1])

parser.add_argument("--batch_size", type=int, default=1024, help="batch size per device for training Unet model", )
parser.add_argument("--eval_batch_size", type=int, default=2048, help="batch size per device for training Unet model", )
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--lr_reg", type=float, default=1e-4, help="learning rate for regression")
parser.add_argument("--wd", type=float, default=1e-2, help="weight decay degree")
parser.add_argument("--drop_rate", type=float, default=0.1, help="drop rate for DownstreamRegressionModel", )
parser.add_argument("--warmup_ratio", type=float, default=0.05, help="warmup ratio for fine-tuning scheduling", )

parser.add_argument("--prefix", type=str, default="../../ext_hdd/hskong/")

parser.add_argument("--finetune", default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument("--pretrain", default=True, type=lambda s: s in ["True", "true", 1])
parser.add_argument("--layer", type=str, default="multiple", choices=["linear", "multiple"])

# KNN related parameters
parser.add_argument("--k", default=100, type=int)
parser.add_argument("--nlist", default=1024, type=int)
parser.add_argument("--nprobe", default=16, type=int)
parser.add_argument("--qbs", default=2048, type=int)

parser.add_argument("--debug", default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument("--seed", default=1004, type=int)
parser.add_argument("--polybert_train", default=False, action="store_true")

parser.add_argument("--wandb", default=False, type=lambda s: s in ["True", "true", 1])
parser.add_argument('--exp_name', default=None, type=str, required=True)
args = parser.parse_args()


def seed_everything(seed=1062):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def data_refactoring(df):
    column_names = ["smiles", "Egc", "Egb", "Eib", "CED", "Ei", "Eea", "nc", "ne", "TSb", "TSy", "epsb", "epsc",
                    "epse_3.0", "YM", "permCH4", "permCO2", "permH2", "permO2", "permN2", "permHe", "Eat", "rho", "LOI",
                    "Xc", "Xe", "Cp", "Td", "Tg", "Tm", "feature"]
    c_data = [[] for _ in range(len(column_names))]

    for idx, c in enumerate(column_names):
        c_data[idx] = list(df[c])

    data = dict()
    for idx, c in enumerate(column_names):
        data[c] = c_data[idx]

    df = pd.DataFrame(data=data, index=range(len(c_data[0])))

    return df


def load_parquet_files(file_list, drop_cols=None):
    chunks = []
    for f in tqdm(file_list):
        df = pd.read_parquet(f, engine='pyarrow')
        if drop_cols:
            df = df.drop(columns=drop_cols, errors='ignore')
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


class DownstreamRegression(nn.Module):
    def __init__(self, polyBERT, tokenizer, drop_rate=0.1):
        super(DownstreamRegression, self).__init__()
        self.PretrainedModel = polyBERT
        self.PretrainedModel.resize_token_embeddings(len(tokenizer))

        if args.layer == "linear":
            self.Regressor = nn.Linear(
                self.PretrainedModel.config.hidden_size, args.num_classes
            )

        else:
            # self.Regressor = nn.Sequential(
            #     nn.Dropout(drop_rate),
            #     nn.Linear(self.PretrainedModel.config.hidden_size, self.PretrainedModel.config.hidden_size),
            #     nn.SiLU(),
            #     nn.Linear(self.PretrainedModel.config.hidden_size, args.num_classes)
            # )

            self.Regressor = nn.Sequential(
                nn.Linear(
                    self.PretrainedModel.config.hidden_size,
                    self.PretrainedModel.config.hidden_size // 2,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.PretrainedModel.config.hidden_size // 2, args.num_classes
                ),
            )

        # freeze pre-trained model
        for param in self.PretrainedModel.parameters():
            param.requires_grad = args.finetune

    def forward(self, input_ids, attention_mask=None):
        outputs = self.PretrainedModel(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.Regressor(logits)
        return output


class PolyDataset(Dataset):
    def __init__(self, df, tokenizer, max_token_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        content = list(self.df.loc[idx])

        smiles = content[0]
        properties = np.array(content[1:-1])
        feature = content[-1]

        return smiles, properties, feature


# class PolyDataset(Dataset):
#     def __init__(self, df, tokenizer, max_token_len=128):
#         self.psmiles = list(df['smiles'].values)  # (N, 600)
#         self.properties = list(df.iloc[:, 1:-1].values)  # (N, 600)
#         self.fingerprints = list(df.iloc[:, -1].values)  # (N, 600)
#
#     def __len__(self):
#         return len(self.psmiles)
#
#     def __getitem__(self, idx):
#         smiles = self.psmiles[idx]
#         properties = np.array(self.properties[idx])
#         feature = self.fingerprints[idx]
#
#         return smiles, properties, feature


def init_train_setting(models, args):
    tabnet, polybert, _ = models

    steps_per_epoch = args.steps
    training_steps = steps_per_epoch * args.epochs
    warmup_steps = int(training_steps * args.warmup_ratio)

    if args.polybert_train:
        optim_params = [
            {"params": polybert.parameters(), "lr": args.lr, "weight_decay": 0.0},
            {"params": tabnet.parameters(), "lr": args.lr_reg, "weight_decay": args.wd},
        ]

    else:
        optim_params = [
            {"params": tabnet.parameters(), "lr": args.lr_reg, "weight_decay": args.wd},
        ]

    optimizer = AdamW(params=optim_params, lr=args.lr)
    criterion = nn.MSELoss()

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)

    scaler = GradScaler()

    return criterion, optimizer, scheduler, scaler


def get_model(args):
    # ------ TabNet ------
    # n_d: width of decision prediction layer (from 8 to 64) default = 8
    # n_a: width of attention embedding (According to the paper n_d=n_a is usually a good choice.) default = 8
    # n_steps: number of steps in the architecture (from 3 to 10) default = 3
    # n_ind: number of independent block in decoder (default = 1)
    # n_shared: Number of shared Gated Linear Units at each step (from 1 to 5)

    # tabnet = TabNet(inp_dim=600, final_out_dim=args.num_classes, n_d=128, n_a=16, n_shared=1, n_ind=4, n_steps=3, relax=1.5, vbs=64).to("cuda")

    # tabnet = TabNet(inp_dim=600, final_out_dim=args.num_classes, n_d=8, n_a=8, n_shared=1, n_ind=1, n_steps=3, relax=1.5, vbs=64).to("cuda")
    # tabnet = TabNet(inp_dim=600, final_out_dim=args.num_classes, n_d=64, n_a=64, n_shared=1, n_ind=3, n_steps=3, relax=1.5, vbs=64).to("cuda")
    tabnet = TabNet(inp_dim=args.num_classes, final_out_dim=600, n_d=64, n_a=64, n_shared=1, n_ind=3, n_steps=3,
                    relax=1.5, vbs=64).to("cuda")

    # ------ PolyBERT ------
    model_name = "kuelumbus/polyBERT"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.pretrain:
        print("==> Loading pretrained model\n")
        polyBERT = AutoModel.from_pretrained(model_name).to("cuda")

    else:
        print("==> Loading only model structure\n")
        model_config = AutoConfig.from_pretrained(model_name)
        polyBERT = AutoModel.from_config(model_config).to("cuda")

    return tabnet, polyBERT, tokenizer


def get_dataset(tokenizer):
    data_path = "./polyone_with_features"
    files = glob(os.path.join(data_path, "*.parquet"))
    # num_train_files = 3 if args.demo else 30

    num_val_files = 5
    num_test_files = 5
    num_train_files = 10

    # num_val_files = int(len(files) * 0.1)
    # num_test_files = int(len(files) * 0.1)
    # num_train_files = int(len(files) * 0.4)

    if args.debug:
        num_train_files = 1
        num_val_files = 1
        num_test_files = 1

    val_files = list(np.random.choice(files, num_val_files, replace=False))
    files = list(set(files) - set(val_files))

    test_files = list(np.random.choice(files, num_test_files, replace=False))
    train_files = list(set(files) - set(test_files))

    train_files = list(np.random.choice(train_files, num_train_files, replace=False))

    # train_df = data_refactoring(load_parquet_files(train_files, drop_cols=["index"]))
    # val_df = data_refactoring(load_parquet_files(val_files, drop_cols=["index"]))
    # test_df = data_refactoring(load_parquet_files(test_files, drop_cols=["index"]))

    # train_df = train_df.drop(columns="index")
    # val_df = val_df.drop(columns="index")
    # test_df = test_df.drop(columns="index")

    train_df = load_parquet_files(train_files, drop_cols=["index", 'level_0'])
    val_df = load_parquet_files(val_files, drop_cols=["index", 'level_0'])
    test_df = load_parquet_files(test_files, drop_cols=["index", 'level_0'])

    train_dataset = PolyDataset(train_df, tokenizer)
    val_dataset = PolyDataset(val_df, tokenizer)
    test_dataset = PolyDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, args.eval_batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, args.eval_batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader


def get_data_generator(train_loader):
    while True:
        try:
            smiles, properties, feature = next(train_iter)
        except:
            train_iter = iter(train_loader)
            smiles, properties, feature = next(train_iter)
        finally:
            smiles, properties, feature = smiles, properties, feature.cuda()

        yield smiles, properties, feature


def embedding_knn_faiss_ivf(models, target_loader=None, query_loader=None, k=100, nlist=512, nprobe=16, query_batch_size=2048):
    # def embedding_knn_faiss_ivf(models, target_loader=None, query_loader=None, k=100, nlist=2048, nprobe=128, query_batch_size=2048):
    """
    Args:
        query_tensor: (Q, D) torch.Tensor, float32, CUDA or CPU
        target_tensor: (T, D) torch.Tensor, float32, CUDA or CPU
        k: top-k neighbors
        nlist: number of IVF clusters (tradeoff: accuracy vs speed)
        nprobe: number of clusters to probe during search
        query_batch_size: size of query batch (for memory efficiency)

    Returns:
        knn_indices: (Q, k) numpy array of nearest neighbor indices
    """

    model, _, _ = models
    model.eval()

    print("\n=> Extract TARGET embeddings")
    # ------------------------------------------------------------------------------------ #
    query_size = len(query_loader.dataset)
    target_size = len(target_loader.dataset) + query_size

    target_embeddings = np.empty((target_size, 600))
    query_embeddings = np.empty((query_size, 600))

    target_total = 0
    query_total = 0

    with torch.no_grad():
        for idx, (_, _, fingerprint) in tqdm(enumerate(target_loader), desc="Target processing"):
            start = target_total
            end = start + len(fingerprint)

            target_embeddings[start:end] = fingerprint.detach().cpu().numpy().astype(np.float32)
            target_total = end

        flag = target_total

        start = target_total
        for idx, (_, properties, fingerprint) in tqdm(enumerate(query_loader), desc="Query processing"):
            start = target_total
            end = start + len(fingerprint)
            target_embeddings[start: end] = fingerprint.detach().cpu().numpy().astype(np.float32)
            target_total = end

            query_start = query_total
            query_end = query_start + len(fingerprint)
            query_embed, _ = model(properties.cuda())
            query_embeddings[query_start: query_end] = query_embed.detach().cpu().numpy().astype(np.float32)
            query_total = query_end

    target_embeddings = target_embeddings.astype(np.float32)
    query_embeddings = query_embeddings.astype(np.float32)

    target_embeddings = np.ascontiguousarray(target_embeddings)
    query_embeddings = np.ascontiguousarray(query_embeddings)

    # Normalize target/query vectors
    target_embeddings = target_embeddings / np.linalg.norm(target_embeddings, axis=1, keepdims=True)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    print("\n==> Target Embedding shape:", target_embeddings.shape)
    print("==> Query Embedding shape:", query_embeddings.shape)
    print("...DONE")
    # ------------------------------------------------------------------------------------ #

    print("\n=> Build FAISS IVF GPU Index")
    Q, D = query_embeddings.shape

    quantizer = faiss.IndexFlatL2(D)
    cpu_index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)
    cpu_index.train(target_embeddings)  # 반드시 train 필요
    cpu_index.add(target_embeddings)

    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.nprobe = nprobe
    print(f"==> Indexed {gpu_index.ntotal} target vectors (nlist={nlist}, nprobe={nprobe})")
    print("...DONE")

    print("\n==> Search QUERY embeddings in batches")
    all_knn_indices = []

    for start in tqdm(range(0, len(query_embeddings), query_batch_size), desc="Searching queries"):
        end = min(start + query_batch_size, len(query_embeddings))
        query_embed = query_embeddings[start:end]

        _, knn_indices = gpu_index.search(query_embed, k)
        all_knn_indices.append(knn_indices)

    indices = np.vstack(all_knn_indices)
    print("...DONE")

    print("\n=> Compute KNN Accuracy (self-inclusion)")
    total_queries = len(query_embeddings)
    count = 0

    for idx, index in enumerate(indices):
        expected_index = flag + idx
        if expected_index in index:
            count += 1

    knn_acc = count / total_queries * 100
    print(f"==> KNN Accuracy (self in top-{k}): {knn_acc:.3f}%")
    print("...DONE\n")

    return knn_acc


def embedding_knn_faiss_ivfpq(models, target_loader=None, query_loader=None, k=100, nlist=1024, nprobe=64, query_batch_size=512, M=30, nbits=8):
    """
    FAISS IVF+PQ KNN search with GPU

    Args:
        models: (model, _, _) tuple
        k: top-k neighbors
        nlist: number of clusters (IVF)
        nprobe: number of clusters to search
        M: number of PQ subquantizers
        nbits: number of bits per subquantizer code (default 8)
    """
    model, _, _ = models
    model.eval()

    # Extract embedding sizes
    query_size = len(query_loader.dataset)
    target_size = len(target_loader.dataset) + query_size

    # Allocate arrays
    target_embeddings = np.empty((target_size, 600), dtype=np.float32)
    query_embeddings = np.empty((query_size, 600), dtype=np.float32)

    target_total = 0
    query_total = 0

    with torch.no_grad():
        for _, (_, _, fingerprint) in tqdm(enumerate(target_loader), desc="Target processing"):
            b = fingerprint.size(0)
            target_embeddings[target_total:target_total + b] = fingerprint.detach().cpu().numpy()
            target_total += b

        flag = target_total

        for _, (_, properties, fingerprint) in tqdm(enumerate(query_loader), desc="Query processing"):
            b = fingerprint.size(0)
            target_embeddings[target_total:target_total + b] = fingerprint.detach().cpu().numpy()
            target_total += b

            query_embed, _ = model(properties.cuda())
            query_embeddings[query_total:query_total + b] = query_embed.detach().cpu().numpy()
            query_total += b

    # Normalize for cosine similarity (optional)
    # target_embeddings = target_embeddings / np.linalg.norm(target_embeddings, axis=1, keepdims=True)
    # query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    print("\n==> Build FAISS IVF+PQ GPU Index")
    D = target_embeddings.shape[1]

    # Create PQ index
    quantizer = faiss.IndexFlatL2(D)  # base quantizer
    cpu_index = faiss.IndexIVFPQ(quantizer, D, nlist, M, nbits)

    train_samples = target_embeddings[:1_000_000].copy()
    cpu_index.train(train_samples)  # train on target embeddings
    cpu_index.add(train_samples)

    # Move to GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.nprobe = nprobe
    gpu_index.useFloat16 = False
    print(f"==> Indexed {gpu_index.ntotal} vectors with IVF+PQ (nlist={nlist}, M={M}, nbits={nbits})")

    print("\n==> Search QUERY embeddings in batches")
    all_knn_indices = []
    for start in tqdm(range(0, query_embeddings.shape[0], query_batch_size), desc="Query Search"):
        end = min(start + query_batch_size, query_embeddings.shape[0])
        query_batch = query_embeddings[start:end]
        _, knn_indices = gpu_index.search(query_batch, k)
        all_knn_indices.append(knn_indices)

    indices = np.vstack(all_knn_indices)

    # Compute self-inclusion recall
    print("\n==> Compute KNN Accuracy (self-inclusion)")
    count = 0
    for i, index in enumerate(indices):
        expected_index = flag + i
        if expected_index in index:
            count += 1

    knn_acc = count / query_embeddings.shape[0] * 100
    print(f"==> KNN Accuracy (self in top-{k}): {knn_acc:.2f}%")

    return knn_acc


def embedding_knn_faiss_ivf_scalar(models, target_loader=None, query_loader=None, k=100, nlist=1024, nprobe=128, query_batch_size=2048, nbits=8):
    model, _, _ = models
    model.eval()

    print("\n=> Extract TARGET & QUERY embeddings")
    query_size = len(query_loader.dataset)
    target_size = len(target_loader.dataset) + query_size

    target_embeddings = np.empty((target_size, 600), dtype=np.float32)
    query_embeddings = np.empty((query_size, 600), dtype=np.float32)

    target_total, query_total = 0, 0

    with torch.no_grad():
        for _, (_, _, fingerprint) in tqdm(enumerate(target_loader), desc="Target processing"):
            b = fingerprint.size(0)
            target_embeddings[target_total:target_total + b] = fingerprint.detach().cpu().numpy()
            target_total += b

        flag = target_total

        for _, (_, properties, fingerprint) in tqdm(enumerate(query_loader), desc="Query processing"):
            b = fingerprint.size(0)
            target_embeddings[target_total:target_total + b] = fingerprint.detach().cpu().numpy()
            target_total += b

            query_embed, _ = model(properties.cuda())
            query_embeddings[query_total:query_total + b] = query_embed.detach().cpu().numpy()
            query_total += b

    # Normalize (optional)
    # target_embeddings = target_embeddings / np.linalg.norm(target_embeddings, axis=1, keepdims=True)
    # query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    print("...DONE")

    print("\n==> Build FAISS IVF + ScalarQuantizer GPU Index")
    D = target_embeddings.shape[1]

    # Step 1: Define ScalarQuantizer
    # sq = faiss.ScalarQuantizer(D, faiss.ScalarQuantizer.QT_8bit)  # or QT_fp16, QT_4bit
    # cpu_index = faiss.IndexIVFScalarQuantizer(quantizer, D, nlist, sq, faiss.METRIC_L2)

    quantizer = faiss.IndexFlatL2(D)

    cpu_index = faiss.IndexIVFScalarQuantizer(
        quantizer, D, nlist,
        faiss.ScalarQuantizer.QT_8bit,  # <- enum으로 직접 넘김
        faiss.METRIC_L2
    )

    train_samples = target_embeddings[:1_000_000].copy()
    cpu_index.train(train_samples)
    cpu_index.add(target_embeddings)

    # Step 2: Move to GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.nprobe = nprobe

    print(f"==> Indexed {gpu_index.ntotal} vectors (nlist={nlist}, nprobe={nprobe}, nbits={nbits})")
    print("...DONE")

    print("\n==> Search QUERY embeddings in batches")
    all_knn_indices = []
    for start in tqdm(range(0, query_embeddings.shape[0], query_batch_size), desc="Searching queries"):
        end = min(start + query_batch_size, query_embeddings.shape[0])
        query_batch = query_embeddings[start:end]
        _, knn_indices = gpu_index.search(query_batch, k)
        all_knn_indices.append(knn_indices)

    indices = np.vstack(all_knn_indices)
    print("...DONE")

    print("\n==> Compute KNN Accuracy (self-inclusion)")
    count = 0
    for i, index in enumerate(indices):
        expected_index = flag + i
        if expected_index in index:
            count += 1

    knn_acc = count / query_embeddings.shape[0] * 100
    print(f"==> KNN Accuracy (self in top-{k}): {knn_acc:.6f}%")

    return knn_acc


def evaluation(models, eval_loader, criterion, scaler):
    print("\n# Evaluation Phase")

    tabnet, polybert, tokenizer = models

    tabnet.eval()
    polybert.eval()
    losses = []

    eval_pred, eval_true = torch.tensor([]), torch.tensor([])
    pbar = tqdm(total=len(eval_loader))
    with torch.no_grad():
        for step, (_, properties, target) in enumerate(eval_loader):
            properties, target = properties.cuda(), target.cuda()

            with autocast(device_type='cuda'):
                output, l = tabnet(properties)
                loss = criterion(output, target.squeeze())

            losses.append(loss.item())

            eval_pred = torch.cat([eval_pred, output.cpu()])
            eval_true = torch.cat([eval_true, target.cpu()])

            txt = (
                f"Eval Step: [{step + 1:>4d}/{len(eval_loader):>4d}]  "
                f"loss: {np.average(losses):>.4f}. "
            )

            pbar.set_description(txt)
            pbar.update()

        eval_loss = np.average(losses)
        r2_eval = r2_score(eval_true.flatten().cpu().detach().numpy(), eval_pred.flatten().cpu().detach().numpy())

        return eval_loss, r2_eval


def train(models, data_generator, criterion, optimizer, scheduler=None, scaler=None, args=None):
    tabnet, polybert, tokenizer = models

    tabnet.train()
    polybert.train()
    losses = []

    global global_step
    pbar = tqdm(total=args.steps)
    train_pred, train_true = torch.tensor([]), torch.tensor([])
    for step, batch_idx in enumerate(range(args.steps)):
        (_, properties, target) = next(data_generator)

        properties, target = properties.cuda(), target.cuda()

        optimizer.zero_grad()
        with autocast(device_type='cuda'):  # mixed precision context
            output, l = tabnet(properties)
            loss = criterion(output, target)

        losses.append(loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_pred = torch.cat([train_pred, output.cpu()])
        train_true = torch.cat([train_true, target.cpu()])

        txt = (
            f"Epoch: [{args.epoch + 1:>4d}/{args.epochs:>4d}]  "
            f"Train Iter: {global_step + 1:3}/{args.steps * args.epochs:4}. lr: {optimizer.param_groups[0]['lr']:>.5f}. "
            f"loss: {np.average(losses):>.4f}. "
        )

        pbar.set_description(txt)
        pbar.update()

        global_step += 1
        if scheduler: scheduler.step()

    r2_train = r2_score(train_true.flatten().cpu().detach().numpy(), train_pred.flatten().cpu().detach().numpy())

    return np.average(losses), r2_train


def main():
    print("==> Get model")
    tabnet, polybert, tokenizer = get_model(args)
    models = (tabnet, polybert, tokenizer)
    print("...DONE\n")

    print("==> init train setting")
    criterion, optimizer, scheduler, scaler = init_train_setting(models, args)
    print("...DONE\n")

    print("==> Get dataset")
    train_loader, val_loader, test_loader = get_dataset(tokenizer)
    data_generator = get_data_generator(train_loader)
    print("...DONE\n")

    if args.wandb:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"TabNetRegression_{args.exp_name}_{timestamp}"
        if args.debug: run_id = f"TabNetRegression_{args.exp_name}_DEBUG_{timestamp}"

        wandb.init(entity='polymer_foundation', project='TabNet', id=run_id, dir=args.prefix)
        wandb.config.update(args.__dict__, allow_val_change=True)

    global global_step
    global_step = 0
    torch.cuda.empty_cache()

    knn_method = embedding_knn_faiss_ivf
    knn_method = embedding_knn_faiss_ivfpq
    knn_method = embedding_knn_faiss_ivf_scalar

    """
    nlist : 보통 데이터 수의 √N ~ N/10 수준이 적당. 7만 기준 √N ≈ 265, N/10 ≈ 7000
    nprobe : 정확도가 중요하면 ↑, 속도가 중요하면 ↓. 일반적으로 nlist의 1~5%
    """

    knn_acc = knn_method(models, target_loader=train_loader, query_loader=val_loader, k=args.k, nlist=args.nlist, nprobe=args.nprobe, query_batch_size=args.qbs)
    print(f"Initial KNN Accuracy: {knn_acc:.5f}\n")

    for args.epoch in range(args.epochs):
        train_loss, r2_train = train(models, data_generator, criterion, optimizer, scheduler, scaler, args)
        print(
            f"[Train] Epoch: [{args.epoch + 1:3d}/{args.epochs:3d}]  RMSE loss: {train_loss:.5f},  R^2: {r2_train:.5f}")

        if args.wandb:
            wandb.log({"train": {"loss": train_loss, "R^2 score": r2_train}}, step=args.epoch)

        if (args.epoch + 1) % args.interval == 0:
            val_loss, r2_val = evaluation(models, val_loader, criterion, scaler)
            knn_acc = knn_method(models, target_loader=train_loader, query_loader=val_loader, k=args.k,
                                 nlist=args.nlist, nprobe=args.nprobe, query_batch_size=args.qbs)
            print(f'[Validation] RMSE loss: {val_loss:.5f}  R^2: {r2_val:.5f}  KNN Accuracy: {knn_acc:.5f}')

            if args.wandb: wandb.log({"validation": {"loss": val_loss, "R^2 score": r2_val, "KNN Accuracy": knn_acc}},
                                     step=args.epoch)

        print("=" * 100, "\n")

    test_loss, r2_test = evaluation(models, test_loader, criterion, scaler)
    knn_acc = knn_method(models, target_loader=train_loader, query_loader=test_loader, k=args.k, nlist=args.nlist,
                         nprobe=args.nprobe, query_batch_size=args.qbs)
    print(
        f'[Test] Epoch: [{args.epoch + 1:3}/{args.epochs:3}]  RMSE loss: {test_loss:.5f}  R^2: {r2_test:.5f}  KNN Accuracy: {knn_acc:.5f}')
    if args.wandb:
        wandb.log({"test": {"loss": test_loss, "R^2 score": r2_test, "KNN Accuarcy": knn_acc}})
        wandb.finish()


if __name__ == "__main__":
    global global_step
    seed_everything(seed=args.seed)

    main()
