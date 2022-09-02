import random
from collections import defaultdict
from typing import DefaultDict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Embedding, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.data_utils import Sequence
from scipy.sparse import csr_matrix

from aprec.api.action import Action
from aprec.api.user_actions import User
from aprec.losses.get_loss import get_loss
from aprec.recommenders.recommender import Recommender
from aprec.utils.item_id import ItemId


class MatrixFactorizationRecommender(Recommender):
    def __init__(
        self,
        embedding_size: int,
        num_epochs: int,
        loss: str,
        batch_size: int,
        regularization: float = 0.0,
        learning_rate: float = 0.001,
    ):
        self.users = ItemId()
        self.items = ItemId()
        self.user_actions: DefaultDict[int, List[int]] = defaultdict(list)
        self.embedding_size = embedding_size
        self.num_epochs = num_epochs
        self.loss = loss
        self.batch_size = batch_size
        self.sigma: float = 1.0
        self.max_positives: int = 40
        self.regularization = regularization
        self.learning_rate = learning_rate

    def name(self) -> str:
        return "MatrixFactorization"

    def add_action(self, action: Action) -> None:
        self.user_actions[self.users.get_id(action.user_id)].append(
            self.items.get_id(action.item_id)
        )

    def rebuild_model(self) -> None:
        loss = get_loss(
            self.loss, self.items.size(), self.batch_size, self.max_positives
        )

        self.model = Sequential()
        self.model.add(
            Embedding(
                self.users.size(),
                self.embedding_size + 1,
                input_length=1,
                embeddings_regularizer=l2(self.regularization),
            )
        )
        self.model.add(Flatten())
        self.model.add(
            Dense(
                self.items.size(),
                kernel_regularizer=l2(self.regularization),
                bias_regularizer=l2(self.regularization),
            )
        )
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=loss)
        data_generator = DataGenerator(
            self.user_actions, self.users.size(), self.items.size(), self.batch_size
        )
        for epoch in range(self.num_epochs):
            print(f"epoch: {epoch}")
            data_generator.shuffle()
            self.model.fit(data_generator)

    def recommend(
        self,
        user_id: Union[str, int],
        limit: int,
        features: Optional[List[str]] = None
    ) -> List[Tuple[Union[str, int], float]]:
        with tf.device("/cpu:0"):
            model_input = np.array([[self.users.get_id(user_id)]])
            predictions = tf.nn.top_k(self.model.predict(model_input), limit)
            result: List[Tuple[Union[str, int], float]] = []
            for item_id, score in zip(predictions.indices[0], predictions.values[0]):
                result.append((self.items.reverse_id(int(item_id)), float(score)))
            return result

    def recommend_batch(self,
                        recommendation_requests: List[Tuple[Union[int, str], List[str]]],
                        limit: int) -> List[List[Tuple[Union[str, int], float]]]:
        model_input = np.array(
            [[self.users.get_id(request[0])] for request in recommendation_requests]
        )
        predictions = tf.nn.top_k(self.model.predict(model_input), limit)
        result: List[List[Tuple[Union[str, int], float]]] = []
        for idx in range(len(recommendation_requests)):
            request_result = []
            for item_id, score in zip(
                predictions.indices[idx][0], predictions.values[idx][0]
            ):
                request_result.append(
                    (self.items.reverse_id(int(item_id)), float(score))
                )
            result.append(request_result)
        return result


class DataGenerator(Sequence):
    full_matrix: csr_matrix
    users: List[int]
    batch_size: int

    def __init__(
        self,
        user_actions: DefaultDict[int, List[int]],
        n_users: int,
        n_items: int,
        batch_size: int
    ) -> None:
        rows: List[int] = []
        cols: List[int] = []

        for user in user_actions:
            for item in user_actions[user]:
                rows.append(user)
                cols.append(item)

        vals = np.ones(len(rows), dtype=float)
        self.full_matrix = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
        self.users = list(range(n_users))
        self.batch_size = batch_size

    def shuffle(self) -> None:
        random.shuffle(self.users)

    def __len__(self) -> int:
        return len(self.users) // self.batch_size

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray]:
        start = self.batch_size * item
        end = self.batch_size * (item + 1)
        users_buf = []
        targets_buf = []
        for i in range(start, end):
            users_buf.append([self.users[i]])
            targets_buf.append(self.full_matrix[self.users[i]].todense())
        users: np.ndarray = np.array(users_buf)
        targets = np.reshape(
            np.array(targets_buf),
            (self.batch_size, self.full_matrix.shape[1])
        )
        return users, targets
