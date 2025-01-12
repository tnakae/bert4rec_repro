from aprec.evaluation.configs.bert4rec_repro_paper.common_benchmark_config import *
from aprec.evaluation.split_actions import LeaveOneOut

DATASET = "BERT4rec.beauty"
N_VAL_USERS = 2048
MAX_TEST_USERS = 40226
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(
    filter_seen=True, filter_recommenders={"our_bert4rec_longer_seq"}
)
