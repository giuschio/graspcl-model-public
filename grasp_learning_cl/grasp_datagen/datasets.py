import json
import numpy as np

from typing import List, Dict, Any, Callable
from copy import deepcopy


def _serialize(matrix: np.ndarray) -> list:
    return matrix.flatten().tolist()


def _deserialize(flat_list: list) -> np.ndarray:
    return np.array(flat_list).reshape((4, 4))


class SceneGraspDataHandler:
    """
    Manage a grasp log persisted as JSON Lines (one JSON object per line).
    Each entry is a dict that includes:
      - grasp
      - res
    We assume grasp['pose'] exists, and is a 7‑element vector [tx,ty,tz,qx,qy,qz,qw]
    """
    def __init__(self, fpath: str):
        """
        Initialize (or create) a grasp log at `fpath`
        """
        self._fpath = fpath
        # ensure file exists for append
        if self._fpath is not None: open(self._fpath, 'a').close()
        # in-memory cache
        self._data: List[Dict[str, Any]] = []

    @staticmethod
    def data_to_file(data: List[Dict[str, Any]], fpath: str, mode: str='w'):
        data_c = deepcopy(data)
        with open(fpath, mode) as f:
            for line in data_c:
                for key in line.keys():
                    if type(line[key]) == np.ndarray: line[key] = _serialize(line[key])
                f.write(json.dumps(line) + '\n')

    @classmethod
    def from_file(cls, fpath: str) -> 'SceneGraspDataHandler':
        """
        Load grasps from an existing JSON Lines file at `fpath`.
        """
        inst = cls.__new__(cls)
        inst._fpath = fpath
        inst._data = []
        with open(fpath, 'r') as f:
            for line in f:
                if line.strip():
                    inst._data.append(json.loads(line))
                    inst._data[-1]['pose'] = _deserialize(inst._data[-1]['pose'])
        return inst

    def append(self, grasp: Dict[str, Any], res: Dict[str, Any]) -> None:
        """
        Append a new grasp dict and result metrics.

        - `grasp` must contain a 'pose' np.ndarray of shape (7,).
        - Other numpy arrays (1D) are serialized as lists.
        """
        # validate pose
        grasp_c, res_c = deepcopy(grasp), deepcopy(res)
        grasp_c['pose'] = _serialize(grasp_c['pose'])
        # serialize all grasp fields
        entry: Dict[str, Any] = {}
        for key, val in grasp_c.items():
            entry[key] = val.tolist() if isinstance(val, np.ndarray) and val.ndim == 1 else val

        # include result metrics
        entry.update(res_c)

        # write line and cache
        if self._fpath is not None:
            with open(self._fpath, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        self._data.append(entry)

    @property
    def data(self) -> List[Dict[str, Any]]:
        """Return raw entries (pose as lists)."""
        return self._data


def filter_data(data: List[dict], key: str, predicate: Callable[[Any], bool]) -> List[dict]:
    return [entry for entry in data if key in entry and predicate(entry[key])]


def get_data_without_key(data: List[dict], key: str) -> List[dict]:
    return [entry for entry in data if key not in entry]


def add_key_to_data(dict_list, key, value):
    for d in dict_list:
        d[key] = value
    return dict_list


def apply_function_to_data(data: List[dict], key: str, func: Callable):
    return [
        {**item, key: func(item[key]) if key in item else item.get(key)}
        for item in data
    ]