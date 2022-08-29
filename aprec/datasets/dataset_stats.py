from argparse import ArgumentParser
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Set, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from aprec.api.action import Action
from aprec.api.user_actions import UserActions

from .datasets_register import DatasetsRegister

MetricsFunction = Callable[
    [UserActions, Set[Union[int, str]], List[int]],
    Union[int, float]
]

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_colwidth", 256)


def num_users(users: UserActions,
              items: Set[Union[int, str]],
              session_lens: List[int]) -> int:
    return len(users)


def num_items(users: UserActions,
              items: Set[Union[int, str]],
              session_lens: List[int]) -> int:
    return len(items)


def num_interactions(users: UserActions,
                     items: Set[Union[int, str]],
                     session_lens: List[int]) -> int:
    return sum(session_lens)


def average_session_len(users: UserActions,
                        items: Set[Union[int, str]],
                        session_lens: List[int]) -> float:
    return float(np.mean(session_lens))


def median_session_len(users: UserActions,
                       items: Set[Union[int, str]],
                       session_lens: List[int]) -> int:
    return int(np.median(session_lens))


def min_session_len(users: UserActions,
                    items: Set[Union[int, str]],
                    session_lens: List[int]) -> int:
    return int(np.min(session_lens))


def max_session_len(users: UserActions,
                    items: Set[Union[int, str]],
                    session_lens: List[int]) -> int:
    return int(np.max(session_lens))


def p80_session_len(users: UserActions,
                    items: Set[Union[int, str]],
                    session_lens: List[int]) -> float:
    return float(np.percentile(session_lens, 80))


def sparsity(users: UserActions,
             items: Set[Union[int, str]],
             session_lens: List[int]) -> float:
    sum_interacted = 0
    for user in users:
        interacted_items = len(set(users[user]))
        sum_interacted += interacted_items
    return 1 - sum_interacted / (len(users) * len(items))


all_metrics: Dict[str, MetricsFunction] = {
    "num_users": num_users,
    "num_items": num_items,
    "num_interactions": num_interactions,
    "average_session_len": average_session_len,
    "median_session_len": median_session_len,
    "min_session_len": min_session_len,
    "max_session_len": max_session_len,
    "p80_session_len": p80_session_len,
    "sparsity": sparsity,
}


def dataset_stats(
    dataset: Iterable[Action], metrics: List[str], dataset_name: Optional[str] = None
) -> Dict[str, Union[int, float, str]]:
    users: UserActions = defaultdict(list)
    item_ids: Set[Union[int, str]] = set()
    for action in dataset:
        users[action.user_id].append(action)
        item_ids.add(action.item_id)
    session_lens = [len(users[user_id]) for user_id in users]
    result: Dict[str, Union[int, float, str]] = {}
    for metric in metrics:
        if metric not in all_metrics:
            raise Exception(f"unknown dataset metric: {metric}")
        else:
            result[metric] = all_metrics[metric](users, item_ids, session_lens)

    if dataset_name is not None:
        result["name"] = dataset_name
    return result


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--datasets",
        required=True,
        help=f"Available Datasets: {','.join(DatasetsRegister().all_datasets())}",
    )
    parser.add_argument(
        "--metrics",
        required=False,
        help=f"Available Columns: {','.join(all_metrics.keys())}",
        default=",".join(all_metrics.keys()),
    )
    parser.add_argument("--latex_table", required=False, default=False)
    args = parser.parse_args()

    metric_names: List[str] = args.metrics.split(",")
    dataset_names: List[str] = args.datasets.split(",")
    for dataset_name in dataset_names:
        if dataset_name not in DatasetsRegister().all_datasets():
            print(f"unknown dataset {dataset_name}")
            exit(1)
    docs = []
    for dataset_name in tqdm(dataset_names):
        dataset: Iterable[Action] = DatasetsRegister()[dataset_name]()
        stats = dataset_stats(dataset, metric_names, dataset_name=dataset_name)
        docs.append(stats)
        del dataset
    df = pd.DataFrame(docs).set_index("name")
    if not args.latex_table:
        print(df)
    else:
        print(df.to_latex())
