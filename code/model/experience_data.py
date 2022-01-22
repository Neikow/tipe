from typing import Union
from model.scope import Scope

# pylint: disable = unsubscriptable-object

DataEntry = dict[str, Union[float, Scope]]


class ExperienceData (dict[str, DataEntry]):
    """Custom `dict` class holding experience data."""
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, key) -> DataEntry:
        val = dict.__getitem__(self, key)
        return val

    def __setitem__(self, __key: str, val: DataEntry) -> None:
        if not isinstance(val, dict):
            raise Exception('The item must be a `dict`.')

        return super().__setitem__(__key, val)

    def getDataKeys(self) -> list[str]:
        """Returs the keys of the"""
        return list(self[list(self.keys())[0]].keys())

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)

    def update(self, *args, **kwargs):
        """Merges `self` with another dict"""
        for key, val in dict(*args, **kwargs).items():
            self[key] = val