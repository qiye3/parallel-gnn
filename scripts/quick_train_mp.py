from parallel.dataloader_mp import create_mp_dataloader
from core.train.train_baseline import train_baseline

loader = create_mp_dataloader(
    "data/processed/train.parquet",
    num_items,
    edge_list,
    cfg.num_neighbors,
    batch_size=256,
    num_workers=4,
)

train_baseline(model, loader, edge_index, optimizer)
