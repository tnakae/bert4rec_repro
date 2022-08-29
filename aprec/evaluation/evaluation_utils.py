from collections import defaultdict
from typing import List

from aprec.api.action import Action
from aprec.api.user_actions import UserActions


def group_by_user(actions: List[Action]) -> UserActions:
    result = defaultdict(list)
    for action in actions:
        result[action.user_id].append(action)
    return result
