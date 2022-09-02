import math
from typing import List, Optional, Tuple, Union

import keras.layers as layers
import numpy as np
import numpy.typing as npt
from keras.models import Sequential
from keras.utils import Sequence
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from aprec.api.action import Action
from aprec.recommenders.recommender import Recommender
from aprec.utils.item_id import ItemId


class GreedyMLP(Recommender):
    def __init__(self,
                 bottleneck_size: int = 32,
                 train_epochs: int = 300) -> None:
        self.users = ItemId()
        self.items = ItemId()
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.vals: List[float] = []
        self.model: Sequential = None
        self.matrix: csr_matrix = None
        self.bottleneck_size = bottleneck_size
        self.train_epochs = train_epochs

    def name(self) -> str:
        return "GreedyMLP"

    def add_action(self, action: Action) -> None:
        row = self.users.get_id(action.user_id)
        col = self.items.get_id(action.item_id)
        self.rows.append(row)
        self.cols.append(col)
        self.vals.append(1.0)

    def rebuild_model(self) -> None:
        self.matrix = csr_matrix((self.vals, (self.rows, self.cols)))
        self.model = self.get_model(self.matrix.shape[1])

        train_data: csr_matrix
        val_data: csr_matrix
        train_data, val_data = train_test_split(self.matrix)

        generator = BatchGenerator(train_data)
        val_generator = BatchGenerator(val_data)

        self.model.fit(
            generator,
            epochs=self.train_epochs,
            validation_data=val_generator
        )

    def get_model(self, n_movies: int) -> Sequential:
        model = Sequential(name="MLP")
        model.add(layers.Input(shape=(n_movies), name="input"))
        model.add(layers.Dropout(0.5, name="input_drouput"))
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
        if self.users.has_item(user_id):
            slice: csr_matrix = self.matrix[self.users.get_id(user_id)]
            user_vec: npt.NDArray[np.float64] = slice.toarray()
        scores: npt.NDArray[np.float64] = self.model.predict(user_vec)[0]
        best_ids: npt.NDArray[np.int64] = np.argsort(scores)[::-1][:limit]
        result = [(self.items.reverse_id(id), float(scores[id])) for id in best_ids]
        return result

    def get_similar_items(self, item_id, limit):
        raise NotImplementedError

    def to_str(self):
        raise NotImplementedError

    def from_str(self):
        raise NotImplementedError


class BatchGenerator(Sequence):
    def __init__(self,
                 matrix: csr_matrix,
                 batch_size: int = 1000) -> None:
        self.matrix: csr_matrix = matrix
        self.batch_size = batch_size
        self.current_position: int = 0
        self.max: int = super().__len__()

    def __len__(self) -> int:
        return math.ceil(self.matrix.shape[0] / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        slice: csr_matrix = self.matrix[
            (idx * self.batch_size):((idx + 1) * self.batch_size)
        ]
        batch: npt.NDArray[np.float64] = slice.toarray()
        return batch, batch

    def __next__(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if self.current_position >= self.max:
            raise StopIteration()
        result = self.__getitem__(self.current_position)
        self.current_position += 1
        return result
