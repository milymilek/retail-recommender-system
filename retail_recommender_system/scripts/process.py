import argparse
from functools import partial
from typing import cast

from retail_recommender_system.data.datasets.base import BaseDataset
from retail_recommender_system.data.loader import DatasetEnum
from retail_recommender_system.logging import init_logger
from retail_recommender_system.utils import parse_enum, set_seed

logger = init_logger(__name__)


def main(args: argparse.Namespace):
    set_seed(args.seed)

    dataset_cls = cast(type[BaseDataset], args.dataset)
    dataset = dataset_cls(base_input=args.input_base, base_output=args.output_base, prefix=args.prefix, ignore_ds=args.ignore_ds)

    dataset.load_base()
    dataset.process_base()
    dataset.save_intermediate()

    print(dataset.data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=partial(parse_enum, cls=DatasetEnum), required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--input_base", type=str, required=False, default=".data")
    parser.add_argument("--output_base", type=str, required=False, default=".data")
    parser.add_argument("--ignore_ds", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
