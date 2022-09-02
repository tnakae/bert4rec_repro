import math
import random
from collections import defaultdict
from typing import DefaultDict, List, Optional, Tuple, Union

import keras.layers as layers
import numpy as np
import numpy.typing as npt
from keras.models import Sequential
from keras.utils import Sequence
from scipy.sparse import csr_matrix

from aprec.api.action import Action
from aprec.recommenders.recommender import Recommender
from aprec.utils.item_id import ItemId

UserHistory = List[Tuple[int, int]]
UserHistoryDict = DefaultDict[int, UserHistory]
UserHistoryList = List[UserHistory]


class GreedyMLPHistorical(Recommender):
    def __init__(self,
                 bottleneck_size: int = 32,
                 train_epochs: int = 300,
                 n_val_users: int = 1000,
                 batch_size: int = 256) -> None:
        self.users = ItemId()
        self.items = ItemId()
        self.user_actions: UserHistoryDict = defaultdict(lambda: [])
        self.model: Sequential = None
        self.matrix: csr_matrix = None
        self.bottleneck_size = bottleneck_size
        self.train_epochs = train_epochs
        self.n_val_users = n_val_users
        self.batch_size = batch_size

    def name(self) -> str:
        return "GreedyMLPHistorical"

    def add_action(self, action: Action) -> None:
        user_id_internal = self.users.get_id(action.user_id)
        action_id_internal = self.items.get_id(action.item_id)
        self.user_actions[user_id_internal].append(
            (action.timestamp, action_id_internal)
        )

    def user_actions_by_id_list(self, id_list: List[int]) -> UserHistoryList:
        result = []
        for user_id in id_list:
            result.append(self.user_actions[user_id])
        return result

    def split_users(self) -> Tuple[UserHistoryList, UserHistoryList]:
        all_user_ids = list(range(0, self.users.size()))
        random.shuffle(all_user_ids)
        val_users = self.user_actions_by_id_list(all_user_ids[:self.n_val_users])
        train_users = self.user_actions_by_id_list(all_user_ids[self.n_val_users:])
        return train_users, val_users

    def sort_actions(self) -> None:
        for user_id in self.user_actions:
            self.user_actions[user_id].sort()

    def rebuild_model(self) -> None:
        self.sort_actions()
        train_users, val_users = self.split_users()
        val_generator = BatchHistoryGenerator(val_users, self.items.size(), self.batch_size)
        self.model = self.get_model(self.items.size())
        for epoch in range(self.train_epochs):
            print(f"epoch: {epoch}")
            generator = BatchHistoryGenerator(train_users, self.items.size(), self.batch_size)
            self.model.fit(generator, validation_data=val_generator)

    def get_model(self, n_movies: int) -> Sequential:
        model = Sequential(name="MLP")
        model.add(layers.Input(shape=(n_movies), name="input"))
        model.add(layers.Dense(256, name="dense1", activation="relu"))
        model.add(layers.Dense(128, name="dense2", activation="relu"))
        model.add(
            layers.Dense(self.bottleneck_size, name="bottleneck", activation="relu")
        )
        model.add(layers.Dense(128, name="dense3", activation="relu"))
        model.add(layers.Dense(256, name="dense4", activation="relu"))
        model.add(layers.Dropout(0.5, name="dropout"))
        model.add(layers.Dense(n_movies, name="output", activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    def recommend(
        self,
        user_id: Union[str, int],
        limit: int,
        features: Optional[List[str]] = None
    ) -> List[Tuple[Union[str, int], float]]:
        vector: npt.NDArray[np.float64] = np.zeros(self.items.size())
        if self.users.has_item(user_id):
            actions = self.user_actions[self.users.get_id(user_id)]
            for action in actions:
                vector[action[1]] = 1
        return self.get_model_predictions(vector, limit)

    def get_model_predictions(
        self,
        vector: npt.NDArray[np.float64],
        limit: int
    ) -> List[Tuple[Union[str, int], float]]:
        scores: npt.NDArray[np.float64] = self.model.predict(vector.reshape(1, self.items.size()))[0]
        best_ids: npt.NDArray[np.int64] = np.argsort(scores)[::-1][:limit]
        result = [(self.items.reverse_id(id), float(scores[id])) for id in best_ids]
        return result

    def recommend_by_items(
        self,
        items_list: List[Union[int, str]],
        limit: int
    ) -> List[Tuple[Union[str, int], float]]:
        vector: npt.NDArray[np.float64] = np.zeros(self.items.size())
        for item in items_list:
            item_id = self.items.get_id(item)
            vector[item_id] = 1
        return self.get_model_predictions(vector, limit)

    def get_similar_items(self, item_id, limit):
        raise NotImplementedError

    def to_str(self):
        raise NotImplementedError

    def from_str(self):
        raise NotImplementedError


class BatchHistoryGenerator(Sequence):
    def __init__(self,
                 user_actions: UserHistoryList,
                 n_items: int,
                 batch_size: int = 256) -> None:
        history, target = BatchHistoryGenerator.split_actions(user_actions)
        self.features_matrix = self.build_matrix(history, n_items)
        self.target_matrix = self.build_matrix(target, n_items)
        self.batch_size = batch_size
        self.current_position: int = 0
        self.max: int = super().__len__()

    @staticmethod
    def build_matrix(user_actions: UserHistoryList,
                     n_items: int) -> csr_matrix:
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []

        for i in range(len(user_actions)):
            for action in user_actions[i]:
                rows.append(i)
                cols.append(action[1])
                vals.append(1.0)

        return csr_matrix((vals, (rows, cols)),
                          shape=(len(user_actions), n_items))

    @staticmethod
    def split_actions(user_actions: UserHistoryList) -> Tuple[UserHistoryList, UserHistoryList]:
        history: UserHistoryList = []
        target: UserHistoryList = []
        for user in user_actions:
            user_history, user_target = BatchHistoryGenerator.split_user(user)
            history.append(user_history)
            target.append(user_target)
        return history, target

    @staticmethod
    def split_user(user: UserHistory) -> Tuple[UserHistory, UserHistory]:
        history_fraction = random.random()
        n_history_actions = int(len(user) * history_fraction)
        history_actions = user[:n_history_actions]
        target_actions = user[n_history_actions:]
        return history_actions, target_actions

    def __len__(self) -> int:
        return math.ceil(self.features_matrix.shape[0] / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        history_slice: csr_matrix = self.features_matrix[
            (idx * self.batch_size):((idx + 1) * self.batch_size)
        ]
        target_slice: csr_matrix = self.target_matrix[
            (idx * self.batch_size):((idx + 1) * self.batch_size)
        ]
        history: npt.NDArray[np.float64] = history_slice.toarray()
        target: npt.NDArray[np.float64] = target_slice.toarray()
        return history, target

    def __next__(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if self.current_position >= self.max:
            raise StopIteration()
        result = self.__getitem__(self.current_position)
        self.current_position += 1
        return result
