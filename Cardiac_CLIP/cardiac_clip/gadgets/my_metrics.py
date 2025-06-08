import numpy as np
import sklearn.metrics as sklm
import torch
from torchmetrics import Metric
from sklearn.preprocessing import label_binarize



class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = logits.argmax(dim=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total

class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class ROCScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("y_trues", default=[], dist_reduce_fx="cat")
        self.add_state("y_scores", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float(),
            target.detach().float(),
        )

        y_true = target
        #y_score = 1 / (1 + torch.exp(-logits))
        y_score=logits
        self.y_trues.append(y_true)
        self.y_scores.append(y_score)


    def compute(self):
        if not isinstance(self.y_trues, list):
            self.y_trues=[self.y_trues]
        if not isinstance(self.y_scores, list):
            self.y_scores=[self.y_scores]
        try:
            score = sklm.roc_auc_score(np.concatenate([y_true.cpu().numpy() for y_true in self.y_trues], axis=0),
                                       np.concatenate([y_score.cpu().numpy() for y_score in self.y_scores], axis=0))
            self.score = torch.tensor(score).to(self.score)
        except ValueError as e:
            #print("ValueError:", e)
            self.score = torch.tensor(0).to(self.score)
            #score = sklm.roc_auc_score(np.concatenate([y_true.cpu().numpy() for y_true in [self.y_trues]], axis=0),
                                       #np.concatenate([y_score.cpu().numpy() for y_score in [self.y_scores]], axis=0))
            #self.score = torch.tensor(score).to(self.score)
        return self.score

class MultiClassROCScore(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state("y_trues", default=[], dist_reduce_fx="cat")
        self.add_state("y_scores", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float(),
            target.detach().long() 
        )

        y_true = target
        y_score = logits.softmax(dim=1)

        self.y_trues.append(y_true)
        self.y_scores.append(y_score)

    def compute(self):
        if not isinstance(self.y_trues, list):
            self.y_trues = [self.y_trues]
        if not isinstance(self.y_scores, list):
            self.y_scores = [self.y_scores]

        try:
            y_true = np.concatenate([y_true.cpu().numpy() for y_true in self.y_trues], axis=0)
            y_score = np.concatenate([y_score.cpu().numpy() for y_score in self.y_scores], axis=0)
            
            classes = [0, 1, 2,3]
            y_true_binarized = label_binarize(y_true, classes=classes)

            
            valid_auc_scores = []
            for class_idx in range(len(classes)):
                y_true_class = y_true_binarized[:, class_idx]
                if len(np.unique(y_true_class)) >= 2:
                    auc = sklm.roc_auc_score(y_true_class, y_score[:, class_idx])
                    valid_auc_scores.append(auc)

            if valid_auc_scores:
                score = np.mean(valid_auc_scores)
            self.score = torch.tensor(score).to(self.score)
        except Exception as e:
            print(e)
            self.score = torch.tensor(0).to(self.score)

        return self.score

class F1Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("y_trues", default=[], dist_reduce_fx="cat")
        self.add_state("y_preds", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float(),
            target.detach().float(),
        )

        y_true = target
        y_score = 1 / (1 + torch.exp(-logits)) > 0.5
        self.y_trues.append(y_true)
        self.y_preds.append(y_score)

    def compute(self):
        try:
            score = sklm.f1_score(np.concatenate([y_true.cpu().numpy() for y_true in self.y_trues], axis=0),
                                  np.concatenate([y_pred.cpu().numpy() for y_pred in self.y_preds], axis=0))
            self.score = torch.tensor(score).to(self.score)
        except ValueError:
            self.score = torch.tensor(0).to(self.score)
        return self.score
