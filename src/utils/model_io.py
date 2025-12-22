import pickle
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def save_agent(agent: Any, path: str, metadata: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_data = {
        'agent': agent,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat(),
        'agent_type': type(agent).__name__,
    }

    with open(path, 'wb') as f:
        pickle.dump(save_data, f)


def load_agent(path: str) -> Tuple[Any, Dict]:
    with open(path, 'rb') as f:
        save_data = pickle.load(f)

    return save_data['agent'], save_data.get('metadata', {})


def save_results(results: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)


def load_results(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def save_trajectories(trajectories: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(trajectories, f)


def load_trajectories(path: str) -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)
