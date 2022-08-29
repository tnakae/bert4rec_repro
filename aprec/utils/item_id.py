from typing import Dict, Union

from collections import Counter


class ItemId:
    def __init__(self) -> None:
        self.straight: Dict[Union[str, int], int] = {}
        self.reverse: Dict[int, Union[str, int]] = {}
        self.counter: Counter = Counter()

    def size(self) -> int:
        return len(self.straight)

    def get_id(self, item_id: Union[str, int]) -> int:
        if item_id not in self.straight:
            self.straight[item_id] = len(self.straight)
            self.reverse[self.straight[item_id]] = item_id
        self.counter[item_id] += 1
        return self.straight[item_id]

    def has_id(self, id: int) -> bool:
        return id in self.reverse

    def has_item(self, item_id: Union[str, int]) -> bool:
        return item_id in self.straight

    def reverse_id(self, id: int) -> Union[str, int]:
        return self.reverse[id]
