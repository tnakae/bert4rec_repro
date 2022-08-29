from typing import DefaultDict, List, Union

from aprec.api.action import Action

UserActions = DefaultDict[Union[str, int], List[Action]]
