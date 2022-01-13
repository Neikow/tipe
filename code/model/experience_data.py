from typing import Union
from model.scope import Scope

# pylint: disable = unsubscriptable-object

DataEntry = dict[str, Union[float, Scope]]


class ExperienceData (dict[str, DataEntry]):
    """Custom `dict` class holding experience data."""
    _data_keys: list[str]

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, key) -> DataEntry:
        val = dict.__getitem__(self, key)
        return val

    def __setitem__(self, __key: str, val: DataEntry) -> None:
        if not isinstance(val, dict):
            raise 'The item must be a `dict`.'

        if not hasattr(self, '_data_keys'):
            self._data_keys = list(val.keys())
        else:
            for key in val.keys():
                if key not in self._data_keys:
                    self._data_keys.append(key)

        return super().__setitem__(__key, val)

    def get_data_keys(self) -> list[str]:
        """Returs the keys of the"""
        if not self._data_keys:
            self._data_keys = []

        return self._data_keys

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)

    def update(self, *args, **kwargs):
        """Merges `self` with another dict"""
        for key, val in dict(*args, **kwargs).items():
            self[key] = val

    def set_data_keys(self, new_keys: list[str]):
        """Overrides the current data keys."""
        self._data_keys = new_keys.copy()
