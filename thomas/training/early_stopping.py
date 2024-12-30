# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import operator


class EarlyStopping:
    def __init__(self, patience=3, mode="max"):
        assert mode in ["min", "max"]
        if mode == "min":
            self.better = operator.lt
        elif mode == "max":
            self.better = operator.gt
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_current_best = False
        self.best_model = None

    def step(self, val_metric, model_id):
        if self.best_score is None or self.better(val_metric, self.best_score):
            self.is_current_best = True
            self.best_score = val_metric
            self.counter = 0
            self.best_model = model_id
        else:
            self.is_current_best = False
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True

    def is_best(self):
        return self.is_current_best

    def is_early_stop(self):
        return self.early_stop

    def get_best_model(self):
        return self.best_model
