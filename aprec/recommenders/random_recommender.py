from typing import List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt

from aprec.api.action import Action
from aprec.recommenders.recommender import Recommender


class RandomRecommender(Recommender):
    def __init__(self):
        self.items_set: Set[Union[str, int]] = set()

    def add_action(self, action: Action):
        self.items_set.add(action.item_id)

    def rebuild_model(self):
        self.items: List[Union[str, int]] = list(self.items_set)

    def recommend(
        self,
        user_id: Union[str, int],
        limit: int,
        features: Optional[List[str]] = None
    ) -> List[Tuple[Union[str, int], float]]:
        recommended_items: npt.NDArray[np.int64] = \
            np.random.choice(self.items, limit, replace=False).astype(int)
        result: List[Tuple[Union[str, int], float]] = []
        current_score = 1.0
        for item in recommended_items:
            result.append((int(item), current_score))
            current_score *= 0.9
        return result
