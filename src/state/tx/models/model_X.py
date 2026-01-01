import logging
from typing import Dict, Optional

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from geomloss import SamplesLoss
import torch.nn.functional as F

from .base import PerturbationModel
from .decoders import FinetuneVCICountsDecoder
from .decoders_nb import NBDecoder, nb_nll
from .utils import build_mlp, get_activation_class, get_transformer_backbone

logger = logging.getLogger(__name__)



def pds_contrastive_loss(pred, true, temperature=0.1):
    """
    pred, true: (B, G)
    """
    # normalize embeddings
    pred = F.normalize(pred, dim=-1)
    true = F.normalize(true, dim=-1)

    # similarity matrix
    logits = torch.matmul(pred, true.T) / temperature  # (B, B)

    labels = torch.arange(pred.size(0), device=pred.device)

    return F.cross_entropy(logits, labels)

def pds_contrastive_ce_loss_label_masked(
                                    pred_expr,
                                    true_expr,
                                    perturbation_onehot,
                                    temperature=0.1,
                                ):
    B = pred_expr.size(0)

    pred = F.normalize(pred_expr, dim=1)
    true = F.normalize(true_expr, dim=1)

    logits = torch.matmul(pred, true.T) / temperature

    same_perturb = (perturbation_onehot @ perturbation_onehot.T) > 0
    mask = ~same_perturb
    mask.fill_diagonal_(True)

    logits = logits.masked_fill(~mask, float("-inf"))

    labels = torch.arange(B, device=pred.device)
    return F.cross_entropy(logits, labels)

def des_loss(pred_delta, true_delta, k=200):
    """
    pred_delta, true_delta: (B, G)
    Computes DES-style loss using top-K DE genes from ground truth
    """
    B, G = pred_delta.shape

    # absolute change for DE selection
    true_abs = torch.abs(true_delta)

    # get top-k DE genes per sample
    topk_idx = torch.topk(true_abs, k=k, dim=1).indices  # (B, k)

    # gather predicted and true deltas
    pred_topk = torch.gather(pred_delta, 1, topk_idx)
    true_topk = torch.gather(true_delta, 1, topk_idx)

    return ((pred_topk - true_topk) ** 2).mean()
    

def fused_vcc_loss(
    pred_expr,
    true_expr,
    control_expr,
    perturbation_onehot,
    w_pds=1,
    w_des=1,
    k_des=200
):
    """
    pred_expr, true_expr, control_expr: (B, G)
    """
    pred_delta = pred_expr - control_expr
    true_delta = true_expr - control_expr

    loss_pds = pds_contrastive_ce_loss_label_masked(pred_expr, true_expr, perturbation_onehot)
    loss_des = des_loss(pred_delta, true_delta, k=k_des)
    
    total_loss = (
        w_pds * loss_pds + 
        w_des * loss_des 
    )
    # print ("loss pds and des: ", loss_pds, loss_des)
    return {
        "total": total_loss,
        "pds": loss_pds,
        "des": loss_des,
    }


class PseudobulkPerturbationModelX(PerturbationModel):
    """
    This model:
      1) Projects basal expression and perturbation encodings into a shared latent space.
      2) Uses an OT-based distributional loss (energy, sinkhorn, etc.) from geomloss.
      3) Enables cells to attend to one another, learning a set-to-set function rather than
      a sample-to-sample single-cell map.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "GPT2",
        transformer_backbone_kwargs: dict = None,
        output_space: str = "gene",
        gene_dim: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            input_dim: dimension of the input expression (e.g. number of genes or embedding dimension).
            hidden_dim: not necessarily used, but required by PerturbationModel signature.
            output_dim: dimension of the output space (genes or latent).
            pert_dim: dimension of perturbation embedding.
            gpt: e.g. "TranslationTransformerSamplesModel".
            model_kwargs: dictionary passed to that model's constructor.
            loss: choice of distributional metric ("sinkhorn", "energy", etc.).
            **kwargs: anything else to pass up to PerturbationModel or not used.
        """
        # Call the parent PerturbationModel constructor
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gene_dim=gene_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,
            output_space=output_space,
            **kwargs,
        )

        # Save or store relevant hyperparams
        self.predict_residual = predict_residual
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.distributional_loss = distributional_loss
        self.cell_sentence_len = self.transformer_backbone_kwargs["n_positions"]
        self.gene_dim = gene_dim
        self.batch_dim = batch_dim
        self.pred_delta = kwargs.get("pred_delta")
        self.aux_loss = kwargs.get("aux_loss")
        # Build the distributional loss from geomloss
        blur = kwargs.get("blur", 0.05)
        loss_name = kwargs.get("loss", "energy")
        if loss_name == "energy":
            self.loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
        elif loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        # Build the underlying neural OT network
        self._build_networks()

        self.batch_encoder = None
        if kwargs.get("batch_encoder", False) and batch_dim is not None:
            self.batch_encoder = build_mlp(
                in_dim=batch_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )

        # if the model is outputting to counts space, apply softplus
        # otherwise its in embedding space and we don't want to
        is_gene_space = kwargs["embed_key"] == "X_hvg" or kwargs["embed_key"] is None
        if kwargs.get("softplus", False) and is_gene_space:
            # actually just set this to a relu for now
            self.relu = torch.nn.ReLU()

        if kwargs.get("nb_decoder", False):
            self.gene_decoder = NBDecoder(
                latent_dim=self.output_dim + (self.batch_dim or 0),
                gene_dim=gene_dim,
                hidden_dims=[512, 512, 512],
                dropout=self.dropout,
            )

        control_pert = kwargs.get("control_pert", "non-targeting")
        if kwargs.get("finetune_vci_decoder", False):
            gene_names = []

            if output_space == "gene":
                # hvg's but for which dataset?
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/datasets/hvg/replogle/jurkat.h5")
                    gene_names = temp.var.index.values
            else:
                assert output_space == "all"
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    # temp = ad.read_h5ad('/scratch/ctc/ML/vci/paper_replogle/jurkat.h5')
                    # gene_names = temp.var.index.values
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/cross_dataset/replogle/jurkat.h5")
                    gene_names = temp.var.index.values

            self.gene_decoder = FinetuneVCICountsDecoder(
                genes=gene_names,
                # latent_dim=self.output_dim + (self.batch_dim or 0),
            )

        print(self)

    def _build_networks(self):
        """
        Here we instantiate the actual GPT2-based model.
        """
        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        # Map the input embedding to the hidden space
        self.basal_encoder = build_mlp(
            in_dim=self.input_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        # self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
        #     self.transformer_backbone_key,
        #     self.transformer_backbone_kwargs,
        # )
        
        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """If needed, define how we embed the raw perturbation input."""
        return self.pert_encoder(pert)

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we embed basal state input, if needed."""
        return self.basal_encoder(expr)

    def forward(self, batch: dict, padded=True) -> torch.Tensor:
        """
        The main forward call. Batch is a flattened sequence of cell sentences,
        which we reshape into sequences of length cell_sentence_len.

        Expects input tensors of shape (B, S, N) where:
        B = batch size
        S = sequence length (cell_sentence_len)
        N = feature dimension

        The `padded` argument here is set to True if the batch is padded. Otherwise, we
        expect a single batch, so that sentences can vary in length across batches.
        """
        if padded:
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
        else:
            # we are inferencing on a single batch, so accept variable length sentences
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)
        
        # Shape: [B, S, hidden_dim]
        pert_embedding = self.encode_perturbation(pert)
        control_cells = self.encode_basal_expression(basal)

        seq_input = pert_embedding + control_cells  # Shape: [B, S, hidden_dim]
        batch_size = seq_input.shape[0]

        if self.batch_encoder is not None:
            if padded:
                batch = batch["batch"].reshape(-1, self.cell_sentence_len, self.batch_dim)
            else:
                batch = batch["batch"].reshape(1, -1, self.batch_dim)

            seq_input = seq_input + self.batch_encoder(batch)  # Shape: [B, S, hidden_dim]

        # take the average across the sequence dimension
        seq_input = seq_input.mean(dim=1, keepdim=True)  # Shape: [B, 1, hidden_dim]

        # forward pass + extract CLS last hidden state
        # res_pred = self.transformer_backbone(inputs_embeds=seq_input).last_hidden_state  # Shape: [B, 1, hidden_dim]
        res_pred = seq_input #self.linear_fc(seq_input)
        out_dim = res_pred.shape[-1]

        # broadcast to the sequence length
        if padded:
            res_pred = res_pred.expand(batch_size, self.cell_sentence_len, out_dim)  # Shape: [B, S, hidden_dim]
        else:
            res_pred = res_pred.expand(1, -1, out_dim)

        # add to basal if predicting residual
        if self.predict_residual:
            # treat the actual prediction as a residual sum to basal
            out_pred = self.project_out(res_pred + control_cells)
        else:
            out_pred = self.project_out(res_pred)

        # apply softplus if specified and we output to HVG space
        is_gene_space = self.hparams["embed_key"] == "X_hvg" or self.hparams["embed_key"] is None
        if self.hparams.get("softplus", False) and is_gene_space:
            out_pred = self.relu(out_pred)

        return out_pred.reshape(-1, self.output_dim)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # Get model predictions (in latent space)
        pred = self.forward(batch, padded=padded)
        ctrl = batch["ctrl_cell_emb"]
        if self.pred_delta:
            pert = batch["pert_cell_emb"]
            target = pert - ctrl
        else:
            target = batch["pert_cell_emb"]
        if self.aux_loss:
            auxiliary_loss = fused_vcc_loss(pred, target, ctrl, batch["pert_emb"])

        if padded:
            pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        else:
            pred = pred.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)


        main_loss = self.loss_fn(pred, target).nanmean()
        if self.aux_loss:
            # print ("MSE loss: ", main_loss, auxiliary_loss["total"])
            main_loss += auxiliary_loss["total"]
        self.log("train_loss", main_loss)
                
        # Process decoder if available
        decoder_loss = None
        total_loss = main_loss

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]
            # Train decoder to map latent predictions to gene space
            latent_preds = pred
            # with torch.no_grad():
            #     latent_preds = pred.detach()  # Detach to prevent gradient flow back to main model

            batch_var = batch["batch"].reshape(latent_preds.shape[0], latent_preds.shape[1], -1)
            # concatenate on the last axis
            if self.batch_dim is not None and not isinstance(self.gene_decoder, FinetuneVCICountsDecoder):
                latent_preds = torch.cat([latent_preds, batch_var], dim=-1)

            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_preds)
                if padded:
                    gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                else:
                    gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = total_loss + 0.1 * decoder_loss

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        pred = self.forward(batch)
        ctrl = batch["ctrl_cell_emb"]
        if self.pred_delta:
            pert = batch["pert_cell_emb"]
            target = pert - ctrl
        else:
            target = batch["pert_cell_emb"]
        if self.aux_loss:
            auxiliary_loss = fused_vcc_loss(pred, target, ctrl, batch["pert_emb"])
        
        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
 
        loss = self.loss_fn(pred, target).mean()
        
        if self.aux_loss:
            loss += auxiliary_loss["total"]
        
        self.log("val_loss", loss)

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]

            # Get model predictions from validation step
            latent_preds = pred

            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_preds)  # verify this is automatically detached

                # Get decoder predictions
                pert_cell_counts_preds = pert_cell_counts_preds.reshape(-1, self.cell_sentence_len, self.gene_dim)
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_dim)
                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log the validation metric
            self.log("decoder_val_loss", decoder_loss)

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pred = self.forward(batch, padded=False)
        ctrl = batch["ctrl_cell_emb"]
        if self.pred_delta:
            pert = batch["pert_cell_emb"]
            target = pert - ctrl
        else:
            target = batch["pert_cell_emb"]
        if self.aux_loss:
            auxiliary_loss = fused_vcc_loss(pred, target, ctrl, batch["pert_emb"])
        pred = pred.reshape(1, -1, self.output_dim)
        target = target.reshape(1, -1, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        if self.aux_loss:
            loss += auxiliary_loss["total"]
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'preds', 'X', 'pert_name', etc.
        """
        latent_output = self.forward(batch, padded=padded)  # shape [B, ...]
        if self.pred_delta:
            ctrl = batch["ctrl_cell_emb"]
            latent_output += ctrl

        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }

        basal_hvg = batch.get("ctrl_cell_counts", None)

        if self.gene_decoder is not None:
            if latent_output.dim() == 2:
                batch_var = batch["batch"].reshape(latent_output.shape[0], -1)
            else:
                batch_var = batch["batch"].reshape(latent_output.shape[0], latent_output.shape[1], -1)
            # concatenate on the last axis
            if self.batch_dim is not None and not isinstance(self.gene_decoder, FinetuneVCICountsDecoder):
                latent_output = torch.cat([latent_output, batch_var], dim=-1)
            if isinstance(self.gene_decoder, NBDecoder):
                mu, _ = self.gene_decoder(latent_output)
                pert_cell_counts_preds = mu
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict
