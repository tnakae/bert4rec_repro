import numpy as np
import tensorflow as tf
from keras import Model
from transformers import DebertaConfig, TFDebertaForMaskedLM

from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import (
    SequentialRecsysModel,
)


class Deberta4Rec(SequentialRecsysModel):
    def __init__(
        self,
        output_layer_activation="linear",
        embedding_size=64,
        max_history_len=100,
        attention_probs_dropout_prob=0.2,
        hidden_act="gelu",
        hidden_dropout_prob=0.2,
        initializer_range=0.02,
        intermediate_size=128,
        num_attention_heads=2,
        num_hidden_layers=3,
        type_vocab_size=2,
    ):
        super().__init__(output_layer_activation, embedding_size, max_history_len)
        self.embedding_size = embedding_size
        self.max_history_length = max_history_len
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.type_vocab_size = type_vocab_size

    def get_model(self):
        deberta_config = DebertaConfig(
            vocab_size=self.num_items + 2,  # +1 for mask item, +1 for padding
            hidden_size=self.embedding_size,
            max_position_embeddings=2 * self.max_history_length,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            initializer_range=self.initializer_range,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            type_vocab_size=self.type_vocab_size,
        )
        return Deberta4RecModel(
            self.batch_size,
            self.output_layer_activation,
            deberta_config,
            self.max_history_length,
        )


class Deberta4RecModel(Model):
    def __init__(
        self,
        batch_size,
        outputput_layer_activation,
        deberta_config,
        sequence_length,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.output_layer_activation = outputput_layer_activation
        self.token_type_ids = tf.constant(
            tf.zeros(shape=(batch_size, deberta_config.max_position_embeddings))
        )
        self.position_ids_for_pred = tf.constant(
            np.array(list(range(1, sequence_length + 1))).reshape(1, sequence_length)
        )
        self.bert = TFDebertaForMaskedLM(deberta_config)

    def call(self, inputs, **kwargs):
        sequences = inputs[0]
        labels = inputs[1]
        positions = inputs[2]
        result = self.bert(sequences, labels=labels, position_ids=positions)
        return result.loss

    def score_all_items(self, inputs):
        sequence = inputs[0]
        result = self.bert(sequence, position_ids=self.position_ids_for_pred).logits[
            :, -1, :-2
        ]
        return result
