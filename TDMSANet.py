# author：TXH
# date：2023/10/23 22:09

import torch
from torch import nn
from torch import Tensor
import math
import torch.nn.functional as F
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from torch.utils import data
import preprocessed


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def rearrange_tensor(self, t):
        b, n, _ = t.shape
        h = self.heads
        d = t.shape[-1] // h
        return t.view(b, n, h, d).transpose(1, 2)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = self.rearrange_tensor(qkv[0]), self.rearrange_tensor(qkv[1]), self.rearrange_tensor(qkv[2])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        b, h, n, d = out.shape
        out = out.transpose(1, 2).reshape(b, n, h * d)
        return self.to_out(out)


class TDMSANet_block(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        MSA(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for layer in self.layers:
            attn = layer[0]
            ff = layer[1]
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ClassificationHead(nn.Module):
    def __init__(self, d_input: int, n_class: int):
        super().__init__()

        self.linear1 = nn.Linear(d_input, d_input)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(d_input, n_class)

    def forward(self, x: torch.Tensor):
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, num_temporal: int, num_band: int, d_model: int):
        super().__init__()
        self.linear = nn.Linear(num_temporal * num_band, d_model)

    def forward(self, x: torch.Tensor):
        bs, ps, ps, t, b = x.shape
        x = x.view(bs, ps * ps, t * b)
        x = self.linear(x)
        return x


class BandEmbedding(nn.Module):
    def __init__(self, num_band: int, d_model: int):
        super().__init__()
        self.linear = nn.Linear(num_band, d_model)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, 0:x.size(1), :]
        return self.dropout(x)


class TemporalPool(nn.Module):
    def __init__(self, mode: str = 'max'):
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor):
        if self.mode == 'max':
            x, _ = torch.max(x, dim=1, keepdim=False)
        elif self.mode == 'mean':
            x = torch.mean(x, dim=1, keepdim=False)
        elif self.mode == 'sum':
            x = torch.sum(x, dim=1, keepdim=False)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, num_temporal: int, d_model: int):
        super().__init__()
        self.linear = nn.Linear(num_temporal, d_model)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        return x


class BandPool(nn.Module):
    def __init__(self, mode: str = 'max'):
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor):
        if self.mode == 'max':
            x, _ = torch.max(x, dim=1, keepdim=False)
        elif self.mode == 'mean':
            x = torch.mean(x, dim=1, keepdim=False)
        elif self.mode == 'sum':
            x = torch.sum(x, dim=1, keepdim=False)
        return x


class TemporalFeatureExtractionModule(nn.Module):
    def __init__(
            self,
            num_band: int,
            d_model: int,
            max_len: int = 5_000,
            nhead: int = 8,
            dim_feedforward: int = 2048,
            nlayers: int = 6,
            dropout: float = 0.1,
            mode: str = 'max',
    ):
        super().__init__()
        self.band_emb = BandEmbedding(num_band, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout, max_len)

        dim_head = d_model // nhead
        self.transformer_encoder = TDMSANet_block(d_model, nlayers, nhead, dim_head, dim_feedforward, dropout)
        self.ln = nn.LayerNorm(d_model)
        self.ln1 = nn.LayerNorm(d_model)

        self.tpool = TemporalPool(mode)

    def forward(self, x: torch.Tensor):
        x = self.band_emb(x)
        x = self.pos_emb(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.transformer_encoder(x)
        x = self.tpool(x)
        x = self.ln(x)
        return x


class SpectralFeatureExtractionModule(nn.Module):
    def __init__(
            self,
            num_temporal: int,
            d_model: int,
            nhead: int = 8,
            dim_feedforward: int = 2048,
            nlayers: int = 6,
            dropout: float = 0.1,
            mode: str = 'max',
    ):
        super().__init__()
        self.t_emb = TemporalEmbedding(num_temporal, d_model)

        dim_head = d_model // nhead
        self.transformer_encoder = TDMSANet_block(d_model, nlayers, nhead, dim_head, dim_feedforward, dropout)

        self.ln = nn.LayerNorm(d_model)
        self.ln1 = nn.LayerNorm(d_model)

        self.bandpool = BandPool(mode)

    def forward(self, x: torch.Tensor):
        x = self.t_emb(x)
        x = self.transformer_encoder(x)
        x = self.bandpool(x)
        x = self.ln(x)
        return x


class SpatialFeatureExtractionModule(nn.Module):
    def __init__(
            self,
            num_band: int,
            num_temporal: int,
            patch_size: int,
            d_model: int,
            max_len: int = 5_000,
            nhead: int = 8,
            dim_feedforward: int = 2048,
            nlayers: int = 6,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.ps = patch_size
        self.patch_emb = PatchEmbedding(num_temporal, num_band, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout, max_len)

        dim_head = d_model // nhead
        self.transformer_encoder = TDMSANet_block(d_model, nlayers, nhead, dim_head, dim_feedforward, dropout)

        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = self.patch_emb(x)
        x = self.pos_emb(x)
        x = self.ln(x)
        x = F.relu(x)
        x = self.transformer_encoder(x)
        x = x[:, self.ps * self.ps // 2, :]
        return x


class TDMSANet(nn.Module):
    def __init__(
            self,
            num_band: int,
            num_temporal: int,
            patch_size: int,
            d_model: int,
            max_len: int = 5_000,
            nhead: int = 8,
            dim_feedforward: int = 128,
            nlayers: int = 2,
            n_class: int = 10,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal_feature = TemporalFeatureExtractionModule(num_band, d_model, max_len, nhead, dim_feedforward, nlayers, dropout)
        self.band_feature = SpectralFeatureExtractionModule(num_temporal, d_model, nhead, dim_feedforward, nlayers, dropout)
        self.spatial_feature = SpatialFeatureExtractionModule(num_band, num_temporal, patch_size, d_model, max_len,
                                                              nhead, dim_feedforward, nlayers, dropout)

        d_input = 3 * d_model
        self.ln = nn.LayerNorm(d_input)

        self.classification = ClassificationHead(d_input, n_class)

    def forward(self, band_feature, temporal_feature, spatial_feature):
        band_feature = torch.jit.fork(self.band_feature, band_feature)
        temporal_feature = torch.jit.fork(self.temporal_feature, temporal_feature)
        spatial_feature = torch.jit.fork(self.spatial_feature, spatial_feature)

        band_feature = torch.jit.wait(band_feature)
        temporal_feature = torch.jit.wait(temporal_feature)
        spatial_feature = torch.jit.wait(spatial_feature)

        x = torch.concat((band_feature, temporal_feature, spatial_feature), dim=1)
        x = self.classification(x)
        return x


def train_with_normal(model, train_loader, criterion, optimizer, device):
    model.train()
    model.to(device)
    for batch_idx, input_list in enumerate(train_loader):
        for idx in range(len(input_list)):
            input_list[idx] = input_list[idx].to(device)
        data_list = input_list[0: -1]
        target = input_list[-1]
        output = model(*data_list)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(model, val_loader, criterion, device):
    model.eval()
    model.to(device)
    val_loss = 0
    total_num_data = len(val_loader.dataset)
    real_labels = torch.empty(total_num_data)
    pred_labels = torch.empty(total_num_data)
    start = 0
    with torch.no_grad():
        for input_list in val_loader:
            for idx in range(len(input_list)):
                input_list[idx] = input_list[idx].to(device)
            data_list = input_list[0: -1]
            target = input_list[-1]
            output = model(*data_list)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            len_target = len(target)
            real_labels[start: start + len_target] = target
            pred_labels[start: start + len_target] = pred
            start += len_target

    val_loss /= len(val_loader.dataset)
    return val_loss, real_labels, pred_labels


def start_train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, max_epoch, device,
                                    patience_early_stopping):
    model.to(device)
    best_metric = 0.0
    best_metric2 = 0.0
    best_recall = 0.0
    best_precision = 0.0
    counter = 0
    t_start = time.perf_counter()
    for epoch in range(max_epoch):
        t1 = time.perf_counter()
        train_with_normal(model, train_loader, criterion, optimizer, device)
        val_loss, label, pred = validate(model, val_loader, criterion, device)
        metric = round(accuracy_score(label, pred) * 100, 2)
        metric2 = round(f1_score(label, pred, average='weighted') * 100, 2)
        recall = round(recall_score(label, pred, average='weighted') * 100, 2)
        precision = round(precision_score(label, pred, average='weighted') * 100, 2)
        t2 = time.perf_counter()

        if metric >= best_metric:
            if metric > best_metric:
                best_metric = metric
                best_metric2 = metric2
                best_recall = recall
                best_precision = precision
                counter = 0
                torch.save(model.state_dict(), 'best_model_state_dict.pth')
                print(
                    f"Epoch: {(epoch + 1):>3d}, Validation Loss: {val_loss:.4f}, Best Metric: {best_metric:.4f}, "
                    f"Current Metric: {metric:.4f}, Best Metric2: {best_metric2:.4f}, Current Metric2: {metric2:.4f}, "
                    f"Not Improve: Improved1, Take Time: {(t2 - t1):.2f}s")
            elif metric == best_metric:
                if metric2 > best_metric2:
                    best_metric2 = metric2
                    best_recall = recall
                    best_precision = precision
                    counter = 0
                    torch.save(model.state_dict(), 'best_model_state_dict.pth')
                    print(
                        f"Epoch: {(epoch + 1):>3d}, Validation Loss: {val_loss:.4f}, Best Metric: {best_metric:.4f}, "
                        f"Current Metric: {metric:.4f}, Best Metric2: {best_metric2:.4f}, Current Metric2: {metric2:.4f}, "
                        f"Not Improve: Improved2, Take Time: {(t2 - t1):.2f}s")
        else:
            counter += 1
            print(
                f"Epoch: {(epoch + 1):>3d}, Validation Loss: {val_loss:.4f}, Best Metric: {best_metric:.4f}, "
                f"Current Metric: {metric:.4f}, Best Metric2: {best_metric2:.4f}, Current Metric2: {metric2:.4f}, "
                f"Not Improve: {counter:>3d}, Take Time: {(t2 - t1):.2f}s")

        if counter >= patience_early_stopping:
            print(
                "Validation metric did not improve for {} epochs. Training stopped.".format(patience_early_stopping))
            break

    print(f'Final Result: Accuracy: {best_metric}%, Recall: {best_recall}%, Precision: {best_precision}%, F1: {best_metric2}%')
    t_end = time.perf_counter()
    print('Total time: {}m {:.2f}s'.format(int((t_end - t_start) // 60), (t_end - t_start) % 60))


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, band_feature, temporal_feature, spatial_feature, label):
        self.band_feature = torch.Tensor(band_feature).to(torch.float32)
        self.temporal_feature = torch.Tensor(temporal_feature).to(torch.float32)
        self.spatial_feature = torch.Tensor(spatial_feature).to(torch.float32)
        self.label = torch.Tensor(label).to(torch.int64)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.band_feature[idx], self.temporal_feature[idx], self.spatial_feature[idx], self.label[idx]


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.init()
        torch.cuda.set_device(0)
    print('{} is available'.format(device))

    s = 'S1'
    batch_size = 64
    lr = 0.001
    patch_size = 17
    head = 8
    d_model = 128

    max_epoch = 500
    patience_early_stopping = 100

    path_image_standard_b = '{}/images/standard_b_final_image.npy'.format(s)
    path_image_standard_t = '{}/images/standard_t_final_image.npy'.format(s)
    path_image_standard_tb = '{}/images/standard_tb_final_image.npy'.format(s)
    path_point_train = '{}/train_points/'.format(s)
    path_point_test = '{}/test_points/'.format(s)

    band_feature_train, _ = preprocessed.point2patch_xy(path_image_standard_b, path_point_train, m=patch_size)
    band_feature_test, _ = preprocessed.point2patch_xy(path_image_standard_b, path_point_test, m=patch_size)

    bs, ps, ps, t, b = band_feature_train.shape
    band_feature_train = band_feature_train.reshape(bs, ps * ps, t, b)
    band_feature_train = band_feature_train[:, ps * ps // 2, ...].reshape(bs, t, b)

    bs, ps, ps, t, b = band_feature_test.shape
    band_feature_test = band_feature_test.reshape(bs, ps * ps, t, b)
    band_feature_test = band_feature_test[:, ps * ps // 2, ...].reshape(bs, t, b)

    temporal_feature_train, _ = preprocessed.point2patch_xy(path_image_standard_t, path_point_train, m=patch_size)
    temporal_feature_test, _ = preprocessed.point2patch_xy(path_image_standard_t, path_point_test, m=patch_size)

    bs, ps, ps, t, b = temporal_feature_train.shape
    temporal_feature_train = temporal_feature_train.reshape(bs, ps * ps, t, b)
    temporal_feature_train = temporal_feature_train[:, ps * ps // 2, ...].reshape(bs, t, b)

    bs, ps, ps, t, b = temporal_feature_test.shape
    temporal_feature_test = temporal_feature_test.reshape(bs, ps * ps, t, b)
    temporal_feature_test = temporal_feature_test[:, ps * ps // 2, ...].reshape(bs, t, b)

    spatial_feature_train, label_train = preprocessed.point2patch_xy(path_image_standard_tb, path_point_train, m=patch_size)
    print(spatial_feature_train.shape)

    spatial_feature_test, label_test = preprocessed.point2patch_xy(path_image_standard_tb, path_point_test, m=patch_size)
    print(spatial_feature_test.shape)

    dataset_train = MyDataset(band_feature_train, temporal_feature_train, spatial_feature_train, label_train)
    dataset_test = MyDataset(band_feature_test, temporal_feature_test, spatial_feature_test, label_test)

    dataloader_train = data.DataLoader(dataset_train, batch_size, shuffle=True)
    dataloader_test = data.DataLoader(dataset_test, batch_size)

    num_band = 0
    n_class = 0
    if s == 'S1':
        num_band = 3
        n_class = 10
    elif s == 'S2':
        num_band = 5
        n_class = 9
    net = TDMSANet(
        num_band=num_band,
        num_temporal=4,
        patch_size=patch_size,
        d_model=d_model,
        max_len=5000,
        nhead=head,
        dim_feedforward=2 * d_model,
        nlayers=2,
        n_class=n_class,
        dropout=0.1,
    ).to(device)

    net = torch.jit.script(net)

    loss_func = nn.CrossEntropyLoss()

    optim = torch.optim.Adam(net.parameters(), lr=lr)

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in net.parameters() if not p.requires_grad)

    print(f'Trainable parameters: {trainable_params}')
    print(f'Non-trainable parameters: {non_trainable_params}')

    start_train_with_early_stopping(
        model=net,
        train_loader=dataloader_train,
        val_loader=dataloader_test,
        criterion=loss_func,
        optimizer=optim,
        max_epoch=max_epoch,
        device=device,
        patience_early_stopping=patience_early_stopping,
    )
