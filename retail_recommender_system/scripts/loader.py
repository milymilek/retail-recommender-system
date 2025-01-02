import time

import polars as pl
from torch.utils.data import DataLoader

from retail_recommender_system.data.loader import *
from retail_recommender_system.evaluation.metrics import *
from retail_recommender_system.models.deepfm import *
from retail_recommender_system.models.mf import *
from retail_recommender_system.models.mfconv import *
from retail_recommender_system.models.mfconv import collate_fn
from retail_recommender_system.utils import batch_dict_to_device, split_by_time

dataset = load_dataset(DataConfig(dataset="hm", prefix="frac_0_01"))
dataset.load()
X_train, X_valid = split_by_time(dataset.data["relations"], date_col=dataset.namings["date"], validation_ratio=0.3)


if __name__ == "__main__":
    val_dataset = MFConvDataset(
        relations=X_valid,
        users=dataset.data["users"],
        items=dataset.data["items"],
        namings=dataset.namings,
        neg_sampl=1,
    )
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=2, collate_fn=collate_fn)
    val_loader_single = DataLoader(val_dataset, batch_size=1024, shuffle=False, collate_fn=collate_fn)
    # eval_dataset = MFConvEvalDataset(base_dataset=val_dataset, user_batch_size=5)

    from tqdm import tqdm

    start_time = time.time()
    for i in tqdm(range(5)):
        next(iter(val_loader))
    end_time = time.time()

    start_time_single = time.time()
    for i in tqdm(range(5)):
        next(iter(val_loader_single))
    end_time_single = time.time()

    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Time taken single: {end_time_single - start_time_single} seconds")
