import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F

from orion_kl_ml.models import layers


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float = 0.0,
        norm: bool = True,
        activation: str = "SiLU",
        **activation_args,
    ):
        super().__init__()
        dims = np.linspace(input_dim, output_dim, n_layers).astype(int)
        layers = []
        for index in range(n_layers - 1):
            layers.append(
                layers.MLPBlock(
                    dims[index],
                    dims[index + 1],
                    dropout,
                    norm,
                    activation,
                    activation_args,
                )
            )
        self.layers = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim


class OrionModel(pl.LightningModule):
    """
    Top layer of abstraction; this system defines an abstract model
    of Orion KL parameters: given a molecule embedding, predict the
    column density, excitation temperature, VLSR, and dV for each component.
    """

    def __init__(self, model, lr: float = 1e-3, weight_decay: float = 0.0):
        super().__init__()
        self.model = model
        # parameter regression
        self.output_layer = nn.Linear(model.output_dim, 4 * 4)
        # output logits for whether the molecule exists in one
        # of the components or not
        self.classifier = nn.Linear(model.output_dim, 4)
        self.param_loss = nn.MSELoss()
        self.class_loss = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor):
        z = self.model(X)
        pred_labels = self.classifier(outputs)
        # reshape into 4 parameters, 4 components per batch
        unweighted_outputs = self.output_layer(z).view(-1, 4, 4)
        ll = pred_labels.sigmoid().unsqueeze(-1)
        weighted_outputs = unweighted_outputs * ll
        return weighted_outputs, pred_labels

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("OrionModel")
        group.add_argument("--lr", type=float, default=1e-3)
        group.add_argument("--weight_decay", type=float, default=0.0)
        return parent_parser

    def step(self, batch, prefix):
        # molecule embedding, regression targets, component labels
        X, true_params, labels = batch
        outputs, pred_labels = self(X)
        loss = self.param_loss(outputs, true_params) + self.class_loss(
            pred_labels, labels
        )
        self.log(f"{prefix}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, "validation")
        return loss
