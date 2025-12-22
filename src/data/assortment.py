import numpy as np
import os
import csv
from typing import Tuple, Optional, List, Dict


class InstacartData:

    def __init__(
        self,
        n_products: int = 50,
        n_departments: int = 10,
        context_dim: int = 10,
        data_dir: Optional[str] = None,
        seed: int = 42,
    ):
        self.n_products = n_products
        self.n_departments = n_departments
        self.context_dim = context_dim
        self.rng = np.random.default_rng(seed)

        if data_dir is None:
            data_dir = os.path.expanduser(
                "~/.cache/kagglehub/datasets/psparks/instacart-market-basket-analysis/versions/1"
            )

        self.order_list = []
        self.base_popularity = None
        self.product_depts = None

        if os.path.exists(data_dir):
            self._load_data(data_dir)

        if not self.order_list:
            raise ValueError(
                "Could not load Instacart data. "
                "Run: kagglehub.dataset_download('psparks/instacart-market-basket-analysis')"
            )

    def _load_data(self, data_dir: str):
        products = {}
        products_file = os.path.join(data_dir, 'products.csv')
        with open(products_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 4:
                    prod_id = int(row[0])
                    dept_id = int(row[3])
                    products[prod_id] = {'department': dept_id}

        orders = {}
        orders_file = os.path.join(data_dir, 'orders.csv')
        with open(orders_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 6 and row[2] == 'train':
                    order_id = int(row[0])
                    dow = int(row[4]) if row[4] else 0
                    hour = int(row[5]) if row[5] else 12
                    orders[order_id] = {'dow': dow, 'hour': hour}

        product_counts = {}
        order_products = {}
        op_file = os.path.join(data_dir, 'order_products__train.csv')
        with open(op_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    order_id = int(row[0])
                    prod_id = int(row[1])
                    product_counts[prod_id] = product_counts.get(prod_id, 0) + 1
                    if order_id not in order_products:
                        order_products[order_id] = []
                    order_products[order_id].append(prod_id)

        sorted_products = sorted(product_counts.items(), key=lambda x: -x[1])
        top_products = [p[0] for p in sorted_products[:self.n_products]]
        self.rng.shuffle(top_products)
        self.product_to_idx = {p: i for i, p in enumerate(top_products)}

        self.product_depts = np.array([
            products.get(p, {}).get('department', 0) % self.n_departments
            for p in top_products
        ])

        total = sum(product_counts.get(p, 0) for p in top_products)
        self.base_popularity = np.array([
            product_counts.get(p, 0) / total for p in top_products
        ])

        for order_id, prods in order_products.items():
            if order_id in orders:
                context = orders[order_id]
                purchased = set(prods) & set(top_products)
                if purchased:
                    self.order_list.append({
                        'dow': context['dow'],
                        'hour': context['hour'],
                        'products': [self.product_to_idx[p] for p in purchased if p in self.product_to_idx]
                    })

        print(f"Loaded Instacart data: {len(top_products)} products, {len(self.order_list)} orders")

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        order = self.rng.choice(self.order_list)

        dow_onehot = np.zeros(7)
        dow_onehot[order['dow']] = 1.0
        hour_norm = (order['hour'] - 12) / 12.0

        context = np.concatenate([
            dow_onehot,
            [hour_norm],
            self.rng.standard_normal(self.context_dim - 8)
        ])

        values = self.base_popularity.copy() * 10

        for prod_idx in order['products']:
            if prod_idx < self.n_products:
                values[prod_idx] += 0.5

        values += self.rng.standard_normal(self.n_products) * 0.1
        values = np.maximum(values, 0.01)

        return context, values

    def get_config(self) -> dict:
        return {
            'n_products': self.n_products,
            'context_dim': self.context_dim,
            'n_departments': self.n_departments,
            'n_orders': len(self.order_list),
        }
