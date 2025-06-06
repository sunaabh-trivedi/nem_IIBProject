import time
import copy
import math
from typing import Any, Dict, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from hydra.utils import get_original_cwd
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchmetrics import MeanMetric

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.data_utils import remove_mean, calculate_rmsd_matrix
from dem.utils.logging_utils import fig_to_image

from .components.clipper import Clipper
from .components.cnf import CNF
from .components.ffjord import FFJORD
from .components.distribution_distances import compute_distribution_distances
from .components.ema import EMAWrapper
from .components.ema import EMA
from .components.lambda_weighter import BaseLambdaWeighter
from .components.mlp import TimeConder
from .components.noise_schedules import BaseNoiseSchedule
from .components.prioritised_replay_buffer import PrioritisedReplayBuffer
from .components.scaling_wrapper import ScalingWrapper
from .components.score_estimator import estimate_grad_Rt, wrap_for_richardsons, estimate_score_tweedie
from .components.score_scaler import BaseScoreScaler
from .components.sde_integration import integrate_sde
from .components.sdes import VEReverseSDE
from .components.ais import ais
from .components.kde import log_unnormalize_kde
from .components.annealing_schedule import AnnealingSchedule


def t_stratified_loss(batch_t, batch_loss, num_bins=5, loss_name=None):
    """Stratify loss by binning t."""
    flat_losses = batch_loss.flatten().detach().cpu().numpy()
    flat_t = batch_t.flatten().detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = "loss"
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin + 1]
        t_range = f"{loss_name} t=[{bin_start:.2f},{bin_end:.2f})"
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


def get_wandb_logger(loggers):
    """Gets the wandb logger if it is the list of loggers otherwise returns None."""
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break

    return wandb_logger


class DEMLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        noise_schedule: BaseNoiseSchedule,
        lambda_weighter: BaseLambdaWeighter,
        buffer: PrioritisedReplayBuffer,
        num_init_samples: int,
        num_estimator_mc_samples: int,
        num_samples_to_generate_per_epoch: int,
        num_samples_to_sample_from_buffer: int,
        num_samples_to_save: int,
        eval_batch_size: int,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        nll_with_cfm: bool,
        nll_with_dem: bool,
        nll_on_buffer: bool,
        logz_with_cfm: bool,
        cfm_sigma: float,
        cfm_prior_std: float,
        use_otcfm: bool,
        nll_integration_method: str,
        use_richardsons: bool,
        compile: bool,
        prioritize_cfm_training_samples: bool = False,
        input_scaling_factor: Optional[float] = None,
        output_scaling_factor: Optional[float] = None,
        clipper: Optional[Clipper] = None,
        score_scaler: Optional[BaseScoreScaler] = None,
        partial_prior=None,
        clipper_gen: Optional[Clipper] = None,
        diffusion_scale=1.0,
        cfm_loss_weight=1.0,
        use_ema=False,
        use_exact_likelihood=False,
        debug_use_train_data=False,
        init_from_prior=False,
        compute_nll_on_train_data=False,
        use_buffer=True,
        tol=1e-5,
        version=1,
        negative_time=False,
        num_negative_time_steps=100,
        ais_steps: int = 5,
        ais_dt: float = 0.1,
        ais_warmup: int = 0,
        ema_beta=0.95,
        ema_steps=0,
        iden_t=False,
        sample_noise=False,
        clean_for_w2=True,
        use_tweedie=False,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param buffer: Buffer of sampled objects
        """
        super().__init__()
        # Seems to slow things down
        # torch.set_float32_matmul_precision('high')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net(energy_function=energy_function)
        self.cfm_net = net(energy_function=energy_function)

        self.EMA = EMA(beta=ema_beta, step_start_ema=ema_steps)
        self.ema_model = copy.deepcopy(self.net).eval().requires_grad_(False)
        
        if use_ema:
            self.net = EMAWrapper(self.net)
            self.cfm_net = EMAWrapper(self.cfm_net)
        if input_scaling_factor is not None or output_scaling_factor is not None:
            self.net = ScalingWrapper(self.net, input_scaling_factor, output_scaling_factor)

            self.cfm_net = ScalingWrapper(
                self.cfm_net, input_scaling_factor, output_scaling_factor
            )

        self.score_scaler = None
        if score_scaler is not None:
            self.score_scaler = self.hparams.score_scaler(noise_schedule)

            self.net = self.score_scaler.wrap_model_for_unscaling(self.net)
            self.cfm_net = self.score_scaler.wrap_model_for_unscaling(self.cfm_net)
        
        
        self.dem_cnf = CNF(
            self.net,
            is_diffusion=True,
            use_exact_likelihood=use_exact_likelihood,
            noise_schedule=noise_schedule,
            method=nll_integration_method,
            num_steps=num_integration_steps,
            atol=tol,
            rtol=tol,
        )
        self.cfm_cnf = CNF(
            self.cfm_net,
            is_diffusion=False,
            use_exact_likelihood=use_exact_likelihood,
            method=nll_integration_method,
            num_steps=num_integration_steps,
            atol=tol,
            rtol=tol,
        )
        '''
        self.dem_cnf = FFJORD(
            self.net,
            trace_method='exact' if use_exact_likelihood else 'hutch',
            num_steps=num_integration_steps
        )
        
        self.cfm_cnf = FFJORD(
            self.cfm_net,
            trace_method='exact' if use_exact_likelihood else 'hutch',
            num_steps=num_integration_steps
        )
        '''

        self.ais_steps = ais_steps
        self.ais_dt = ais_dt
        self.ais_warmup = ais_warmup
        
        self.nll_with_cfm = nll_with_cfm
        self.nll_with_dem = nll_with_dem
        self.nll_on_buffer = nll_on_buffer
        self.logz_with_cfm = logz_with_cfm
        self.cfm_prior_std = cfm_prior_std
        self.compute_nll_on_train_data = compute_nll_on_train_data

        flow_matcher = ConditionalFlowMatcher
        if use_otcfm:
            flow_matcher = ExactOptimalTransportConditionalFlowMatcher

        self.cfm_sigma = cfm_sigma
        self.conditional_flow_matcher = flow_matcher(sigma=cfm_sigma)

        self.nll_integration_method = nll_integration_method

        self.energy_function = energy_function
        self.noise_schedule = noise_schedule
        self.buffer = buffer
        self.dim = self.energy_function.dimensionality

        if not self.hparams.debug_use_train_data:
            self.reverse_sde = VEReverseSDE(self.net, self.noise_schedule, self.energy_function)
        else:
            self.reverse_sde = VEReverseSDE(self.cfm_net, self.noise_schedule)
        
        self.use_tweedie = use_tweedie
        if not self.use_tweedie:
            grad_fxn = estimate_grad_Rt
            if use_richardsons:
                grad_fxn = wrap_for_richardsons(grad_fxn)
        else:
            grad_fxn = estimate_score_tweedie
            if use_richardsons:
                grad_fxn = wrap_for_richardsons(grad_fxn)

        self.clipper = clipper
        self.clipped_grad_fxn = self.clipper.wrap_grad_fxn(grad_fxn)

        self.dem_train_loss = MeanMetric()
        self.cfm_train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_nll_logdetjac = MeanMetric()
        self.test_nll_logdetjac = MeanMetric()
        self.val_nll_log_p_1 = MeanMetric()
        self.test_nll_log_p_1 = MeanMetric()
        self.val_nll = MeanMetric()
        self.test_nll = MeanMetric()
        self.val_nfe = MeanMetric()
        self.test_nfe = MeanMetric()
        self.val_energy_w2 = MeanMetric()
        self.val_dist_w2 = MeanMetric()
        self.val_dist_total_var = MeanMetric()

        self.val_dem_nll_logdetjac = MeanMetric()
        self.test_dem_nll_logdetjac = MeanMetric()
        self.val_dem_nll_log_p_1 = MeanMetric()
        self.test_dem_nll_log_p_1 = MeanMetric()
        self.val_dem_nll = MeanMetric()
        self.test_dem_nll = MeanMetric()
        self.val_dem_nfe = MeanMetric()
        self.test_dem_nfe = MeanMetric()
        self.val_dem_logz = MeanMetric()
        self.val_logz = MeanMetric()
        self.test_dem_logz = MeanMetric()
        self.test_logz = MeanMetric()

        self.val_buffer_nll_logdetjac = MeanMetric()
        self.val_buffer_nll_log_p_1 = MeanMetric()
        self.val_buffer_nll = MeanMetric()
        self.val_buffer_nfe = MeanMetric()
        self.val_buffer_logz = MeanMetric()
        self.test_buffer_nll_logdetjac = MeanMetric()
        self.test_buffer_nll_log_p_1 = MeanMetric()
        self.test_buffer_nll = MeanMetric()
        self.test_buffer_nfe = MeanMetric()
        self.test_buffer_logz = MeanMetric()

        self.val_train_nll_logdetjac = MeanMetric()
        self.val_train_nll_log_p_1 = MeanMetric()
        self.val_train_nll = MeanMetric()
        self.val_train_nfe = MeanMetric()
        self.val_train_logz = MeanMetric()
        self.test_train_nll_logdetjac = MeanMetric()
        self.test_train_nll_log_p_1 = MeanMetric()
        self.test_train_nll = MeanMetric()
        self.test_train_nfe = MeanMetric()
        self.test_train_logz = MeanMetric()

        self.num_init_samples = num_init_samples
        self.num_estimator_mc_samples = num_estimator_mc_samples
        self.num_samples_to_generate_per_epoch = num_samples_to_generate_per_epoch
        self.num_samples_to_sample_from_buffer = num_samples_to_sample_from_buffer
        self.num_integration_steps = num_integration_steps
        self.num_samples_to_save = num_samples_to_save
        self.eval_batch_size = eval_batch_size

        self.prioritize_cfm_training_samples = prioritize_cfm_training_samples
        self.lambda_weighter = self.hparams.lambda_weighter(self.noise_schedule)

        self.last_samples = None
        self.last_energies = None
        self.eval_step_outputs = []

        self.partial_prior = partial_prior

        self.clipper_gen = clipper_gen

        self.diffusion_scale = diffusion_scale
        self.init_from_prior = init_from_prior
        self.iden_t = iden_t
        self.sample_noise = sample_noise
        self.clean_for_w2 = clean_for_w2
        
        self.iter_num = 0

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(t, x)

    def get_cfm_loss(self, samples: torch.Tensor) -> torch.Tensor:
        x0 = self.cfm_prior.sample(self.num_samples_to_sample_from_buffer)
        x1 = samples

        t, xt, ut = self.conditional_flow_matcher.sample_location_and_conditional_flow(x0, x1)

        if self.energy_function.is_molecule and self.cfm_sigma != 0:
            xt = remove_mean(
                xt, self.energy_function.n_particles, self.energy_function.n_spatial_dim
            )

        vt = self.cfm_net(t, xt)
        loss = (vt - ut).pow(2).mean(dim=-1)

        #if self.energy_function.normalization_max is not None:
        #   loss = loss / (self.energy_function.normalization_max ** 2)

        return loss

    def should_train_cfm(self, batch_idx: int) -> bool:
        return self.nll_with_cfm or self.hparams.debug_use_train_data

    def get_score_loss(
        self, times: torch.Tensor, samples: torch.Tensor, noised_samples: torch.Tensor
    ) -> torch.Tensor:
        predicted_score = self.forward(times, noised_samples)

        true_score = -(noised_samples - samples) / (
            self.noise_schedule.h(times).unsqueeze(1) + 1e-4
        )
        error_norms = (predicted_score - true_score).pow(2).mean(-1)
        return error_norms

    def buffer_score_estimator(self, times: torch.Tensor, samples: torch.Tensor):
        xt = samples.clone().detach()
        buffer_neg_energy = self.buffer.buffer.energy.clone().detach()
        buffer_data = self.buffer.buffer.x.clone().detach()
        vars = self.noise_schedule.h(times).unsqueeze(-1)
        log_gauassian_t = -0.5 * (xt.unsqueeze(1) - buffer_data.unsqueeze(0)).pow(2).sum(-1) / vars
        log_term = buffer_neg_energy.unsqueeze(0) + log_gauassian_t - log_unnormalize_kde(buffer_data, buffer_data).unsqueeze(0)
        weights = torch.softmax(log_term, dim=1).unsqueeze(-1)
        log_gaussian_t_score = -(xt.unsqueeze(1) - buffer_data.unsqueeze(0)) / vars.unsqueeze(-1)
        score_est = weights * log_gaussian_t_score
        # logsumexp = torch.logsumexp(log_term, dim=1)
        # score_est = torch.autograd.grad(logsumexp.sum(), inputs=xt, create_graph=False)[0]
        # xt.requires_grad_(False)
        return score_est.sum(dim=1)

    def get_loss(self, times: torch.Tensor, samples: torch.Tensor, clean_samples: torch.Tensor, train=False) -> torch.Tensor:
        
        if train:
            self.iter_num += 1
        #clean samples is a placeholder for training on t=0 as regularizer
        if self.ais_steps == 0 or self.iter_num > self.ais_warmup:
            if not self.use_tweedie:
                # estimated_score = estimate_grad_Rt(
                #     times,
                #     samples,
                #     self.energy_function,
                #     self.noise_schedule,
                #     num_mc_samples=self.num_estimator_mc_samples,
                # )
                estimated_score = self.buffer_score_estimator(
                    times,
                    samples
                )
            else:
                estimated_score = estimate_score_tweedie(
                    times,
                    samples,
                    self.energy_function,
                    self.noise_schedule,
                    num_mc_samples=self.num_estimator_mc_samples,
                )
        else:
            estimated_score = ais(
                samples,
                times,
                self.num_estimator_mc_samples,
                self.ais_steps,
                self.noise_schedule,
                self.energy_function,
                dt=self.ais_dt,
            )

        if self.clipper is not None and self.clipper.should_clip_scores:
            if self.energy_function.is_molecule:
                estimated_score = estimated_score.reshape(
                    -1,
                    self.energy_function.n_particles,
                    self.energy_function.n_spatial_dim,
                )

            estimated_score = self.clipper.clip_scores(estimated_score)

            if self.energy_function.is_molecule:
                estimated_score = estimated_score.reshape(-1, self.energy_function.dimensionality)

        if self.score_scaler is not None:
            estimated_score = self.score_scaler.scale_target_score(estimated_score, times)

        predicted_score = self.forward(times, samples)

        error_norms = (predicted_score - estimated_score).pow(2).mean(-1)

        return error_norms * self.lambda_weighter(times)

    def training_step(self, batch, batch_idx):
        loss = 0.0
        if not self.hparams.debug_use_train_data:
            if self.hparams.use_buffer:
                iter_samples, _, _ = self.buffer.sample(self.num_samples_to_sample_from_buffer)
            else:
                iter_samples = self.prior.sample(self.num_samples_to_sample_from_buffer)
                # Uncomment for SM
                # iter_samples = self.energy_function.sample_train_set(self.num_samples_to_sample_from_buffer)

            if isinstance(self.time_schedule, AnnealingSchedule):
                current_epoch = self.current_epoch
                total_epochs = self.time_schedule.num_epochs_to_uniform
                self.time_schedule.anneal_factor = min(current_epoch/total_epochs, 1.0)

                times = self.time_schedule.sample_t(self.num_samples_to_sample_from_buffer).to(iter_samples.device)
            else:
                times = torch.rand(
                    (self.num_samples_to_sample_from_buffer,), device=iter_samples.device
                )

            self.log("time_distribution/mean", times.mean(), on_step=False, on_epoch=True)
            self.log("time_distribution/std", times.std(), on_step=False, on_epoch=True)

            #use this for identical times in one batch
            if self.iden_t:
                if isinstance(self.time_schedule, AnnealingSchedule):
                    times = self.time_schedule.sample_iden_t(self.num_samples_to_sample_from_buffer).to(iter_samples.device)
                else:
                    t = torch.rand([])
                    times = torch.zeros_like(times) + t
                    
            if self.sample_noise:
                noise_h = times ** 2 * self.noise_schedule.h(1)
                noised_samples = iter_samples + (
                    torch.randn_like(iter_samples) * noise_h.sqrt().unsqueeze(-1)
                )
                times = self.noise_schedule.h_to_t(noise_h)
            
            else:
                self.log("time_distribution/noise", self.noise_schedule.h(times).sqrt().mean(), on_step=False, on_epoch=True)
                noised_samples = iter_samples + (
                    torch.randn_like(iter_samples) * self.noise_schedule.h(times).sqrt().unsqueeze(-1)
                )
                # noised_samples = iter_samples + torch.sqrt(times).unsqueeze(-1)*(torch.randn_like(iter_samples).to(iter_samples.device)) # DPS forward process 

            if self.energy_function.is_molecule:
                noised_samples = remove_mean(
                    noised_samples,
                    self.energy_function.n_particles,
                    self.energy_function.n_spatial_dim,
                )

            dem_loss = self.get_loss(times, noised_samples, iter_samples, train=True)
            # Uncomment for SM
            # dem_loss = self.get_score_loss(times, iter_samples, noised_samples)
            #self.log_dict(
            #    t_stratified_loss(times, dem_loss, loss_name="train/stratified/dem_loss")
            #)
            dem_loss = dem_loss.mean()
            loss = loss + dem_loss

            # update and log metrics
            self.dem_train_loss(dem_loss)
            self.log(
                "train/dem_loss",
                self.dem_train_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.should_train_cfm(batch_idx):
            if self.hparams.debug_use_train_data:
                cfm_samples = self.energy_function.sample_train_set(
                    self.num_samples_to_sample_from_buffer
                )
                times = torch.rand(
                    (self.num_samples_to_sample_from_buffer,), device=cfm_samples.device
                )
            else:
                cfm_samples, _, _ = self.buffer.sample(
                    self.num_samples_to_sample_from_buffer,
                    prioritize=self.prioritize_cfm_training_samples,
                )

            cfm_loss = self.get_cfm_loss(self.energy_function.normalize(cfm_samples))
            self.log_dict(
                t_stratified_loss(times, cfm_loss, loss_name="train/stratified/cfm_loss")
            )
            cfm_loss = cfm_loss.mean()
            self.cfm_train_loss(cfm_loss)
            self.log(
                "train/cfm_loss",
                self.cfm_train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            loss = loss + self.hparams.cfm_loss_weight * cfm_loss
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        if self.hparams.use_ema:
            self.net.update_ema()
            if self.should_train_cfm(batch_idx):
                self.cfm_net.update_ema()
    
    @torch.no_grad()
    def generate_samples(
        self,
        reverse_sde: VEReverseSDE = None,
        num_samples: Optional[int] = None,
        return_full_trajectory: bool = False,
        diffusion_scale=1.0,
        negative_time=False,
    ) -> torch.Tensor:
        num_samples = num_samples or self.num_samples_to_generate_per_epoch

        samples = self.prior.sample(num_samples)
        #self.EMA.step_ema(self.ema_model, self.net)
        return self.integrate(
            reverse_sde=reverse_sde,
            samples=samples,
            reverse_time=True,
            return_full_trajectory=return_full_trajectory,
            diffusion_scale=diffusion_scale,
            negative_time=negative_time,
        )

    def integrate(
        self,
        reverse_sde: VEReverseSDE = None,
        samples: torch.Tensor = None,
        reverse_time=True,
        return_full_trajectory=False,
        diffusion_scale=1.0,
        no_grad=True,
        negative_time=False,
    ) -> torch.Tensor:
        trajectory = integrate_sde(
            reverse_sde or self.reverse_sde,
            samples,
            self.num_integration_steps,
            self.energy_function,
            diffusion_scale=diffusion_scale,
            reverse_time=reverse_time,
            no_grad=no_grad,
            negative_time=negative_time,
            num_negative_time_steps=self.hparams.num_negative_time_steps,
        )
        if return_full_trajectory:
            return trajectory
        
        return trajectory[-1]

    @torch.no_grad()
    def compute_nll(
        self,
        cnf,
        prior,
        samples: torch.Tensor,
    ):
        
        aug_samples = torch.cat(
            [samples, torch.zeros(samples.shape[0], 1, device=samples.device)], dim=-1
        )
        aug_output = cnf.integrate(aug_samples)[-1]
        x_1, logdetjac = aug_output[..., :-1], aug_output[..., -1]
        if not cnf.is_diffusion:
            logdetjac = -logdetjac
        log_p_1 = prior.log_prob(x_1)
        log_p_0 = log_p_1 + logdetjac
        nll = -log_p_0
        return nll, x_1#, logdetjac, log_p_1
        '''
        z, delta_logp, reg_term = cnf.forward(samples)
        nll = - (prior.log_prob(z) + delta_logp.view(-1))
        return nll, z
        '''

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.EMA.step_ema(self.ema_model, self.net)
        if self.clipper_gen is not None:
            reverse_sde = VEReverseSDE(
                self.clipper_gen.wrap_grad_fxn(self.ema_model), 
                self.noise_schedule, self.energy_function
            )
            self.last_samples = self.generate_samples(
                reverse_sde=reverse_sde, diffusion_scale=self.diffusion_scale
            )
            self.last_energies = self.energy_function(self.last_samples)
        else:
            reverse_sde = VEReverseSDE(
                self.ema_model, self.noise_schedule, self.energy_function
            )
            self.last_samples = self.generate_samples(
                reverse_sde=reverse_sde, diffusion_scale=self.diffusion_scale
            )
            self.last_energies = self.energy_function(self.last_samples)
        self.buffer.add(self.last_samples, self.last_energies.sum(-1))
        prefix = "val"

        
        self._log_energy_w2(prefix=prefix)
        self._log_data_w2(prefix=prefix)
        
        if self.energy_function.is_molecule:
            self._log_dist_w2(prefix=prefix)
            self._log_dist_total_var(prefix=prefix)
        elif self.energy_function.dimensionality <= 2:
            self._log_data_total_var(prefix=prefix)            

    def _log_energy_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
            if self.clean_for_w2:
                generated_energies = self.energy_function(generated_samples).sum(-1)
                self.log("test/max_energy", generated_energies.max(), on_step=False, on_epoch=True)
                valid_indices = generated_energies > -100
                generated_samples = generated_samples[valid_indices]
                generated_energies = generated_energies[valid_indices]
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, generated_energies = self.buffer.get_last_n_inserted(self.eval_batch_size)    
        energies = self.energy_function(self.energy_function.normalize(data_set)).sum(-1)

        # Ensure both arrays have the same length
        min_length = min(len(energies), len(generated_energies))
        energies = np.random.choice(energies.cpu().numpy(), min_length, replace=False)
        generated_energies = np.random.choice(generated_energies.cpu().numpy(), min_length, replace=False)

        # Compute energy_w2
        energy_w2 = pot.emd2_1d(energies, generated_energies)
        
        self.log(
            f"{prefix}/energy_w2",
            self.val_energy_w2(energy_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    
    def _log_data_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
            if self.clean_for_w2:
                generated_energies = self.energy_function(generated_samples).sum(-1)
                generated_samples = generated_samples[generated_energies > -100]

        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)
        
        generated_samples = self.energy_function.unnormalize(generated_samples)
        if self.energy_function.is_molecule:
            distance_matrix = calculate_rmsd_matrix(data_set.view(-1, 
                                                                  self.energy_function.n_particles,
                                                                  self.energy_function.n_spatial_dim),
                                                    generated_samples.view(-1, 
                                                                  self.energy_function.n_particles,
                                                                  self.energy_function.n_spatial_dim)).cpu().numpy()
        else:
            distance_matrix = pot.dist(data_set.cpu().numpy(), generated_samples.cpu().numpy(), metric='euclidean')
        src, dist = np.ones(len(data_set)) / len(data_set), np.ones(len(generated_samples)) / len(generated_samples)
        G = pot.emd(src, dist, distance_matrix)
        w2_dist = np.sum(G * distance_matrix) / G.sum()
        w2_dist = torch.tensor(w2_dist, device=data_set.device)
        self.log(
            f"{prefix}/data_w2",
            self.val_energy_w2(w2_dist),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _log_dist_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)

        dist_w2 = pot.emd2_1d(
            self.energy_function.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
            self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1),
        )
        self.log(
            f"{prefix}/dist_w2",
            self.val_dist_w2(dist_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _log_dist_total_var(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)

        generated_samples_dists = (
            self.energy_function.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
        )
        data_set_dists = self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1)

        H_data_set, x_data_set = np.histogram(data_set_dists, bins=200)
        H_generated_samples, _ = np.histogram(generated_samples_dists, bins=(x_data_set))
        total_var = (
            0.5
            * np.abs(
                H_data_set / H_data_set.sum() - H_generated_samples / H_generated_samples.sum()
            ).sum()
        )

        self.log(
            f"{prefix}/dist_total_var",
            self.val_dist_total_var(total_var),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
    
    def _log_data_total_var(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)
        
        bins = (200, ) * self.energy_function.dimensionality
        generated_samples = self.energy_function.unnormalize(generated_samples)
        all_data = torch.cat([data_set, generated_samples], dim=0)
        min_vals, _ = all_data.min(dim=0)
        max_vals, _ = all_data.max(dim=0)
        ranges = tuple((min_vals[i].item(), max_vals[i].item()) for i in range(self.energy_function.dimensionality))  # tuple of (min, max) for each dimension
        ranges = tuple(item for subtuple in ranges for item in subtuple)
        hist_p, _ = torch.histogramdd(data_set.cpu(), bins=bins, range=ranges)
        hist_q, _ = torch.histogramdd(generated_samples.cpu(), bins=bins, range=ranges)
        
        p_dist = hist_p / hist_p.sum()
        q_dist = hist_q / hist_q.sum()
        
        total_var = 0.5 * torch.abs(p_dist - q_dist).sum()
        self.log(
            f"{prefix}/data_total_var",
            self.val_dist_total_var(total_var),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        

    def compute_log_z(self, cnf, prior, samples, prefix, name):
        nll, _ = self.compute_nll(cnf, prior, samples)
        # energy function will unnormalize the samples itself
        logz = self.energy_function(samples) + nll
        logz_metric = getattr(self, f"{prefix}_{name}logz")
        logz_metric.update(logz)
        self.log(
            f"{prefix}/{name}logz",
            logz_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
    
    def compute_ess(self, cnf, prior, samples, prefix, name):
        nll, z = self.compute_nll(cnf, prior, samples)
        likelihood = torch.exp(-nll)
        energies = self.energy_function(samples)

        w = torch.exp(energies) / (likelihood + 1e-4)
        w = w / (w.sum() + 1e-4)
        n = torch.tensor(energies.shape[0], device=w.device, dtype=w.dtype)
        self.log(
            f"{prefix}/{name}ess",
            self.val_energy_w2((1/n) / (w ** 2).sum()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def compute_and_log_nll(self, cnf, prior, samples, prefix, name):
        cnf.nfe = 0.0
        nll, forwards_samples = self.compute_nll(cnf, prior, samples)
        nfe_metric = getattr(self, f"{prefix}_{name}nfe")
        nll_metric = getattr(self, f"{prefix}_{name}nll")


        nll_metric.update(nll)

        self.log_dict(
            {
                f"{prefix}/{name}_nfe": nfe_metric,
                # f"{prefix}/{name}logz": logz_metric,
            },
            on_epoch=True,
        )
        self.log(
            f"{prefix}/{name}nll",
            nll_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return forwards_samples

    def eval_step(self, prefix: str, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single eval step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if prefix == "test":
            batch = self.energy_function.sample_test_set(self.eval_batch_size)
        elif prefix == "val":
            batch = self.energy_function.sample_val_set(self.eval_batch_size)

        batch = self.energy_function.normalize(batch)
        backwards_samples = self.last_samples

        # generate samples noise --> data if needed
        if backwards_samples is None or self.eval_batch_size > len(backwards_samples):
            backwards_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )

        # sample eval_batch_size from generated samples from dem to match dimensions
        # required for distribution metrics
        if len(backwards_samples) != self.eval_batch_size:
            indices = torch.randperm(len(backwards_samples))[: self.eval_batch_size]
            backwards_samples = backwards_samples[indices]

        if batch is None:
            print("Warning batch is None skipping eval")
            self.eval_step_outputs.append({"gen_0": backwards_samples})
            return

        times = torch.rand((self.eval_batch_size,), device=batch.device)

        noised_batch = batch + (
            torch.randn_like(batch) * self.noise_schedule.h(times).sqrt().unsqueeze(-1)
        )

        if self.energy_function.is_molecule:
            noised_batch = remove_mean(
                noised_batch,
                self.energy_function.n_particles,
                self.energy_function.n_spatial_dim,
            )

        loss = self.get_loss(times, noised_batch, batch).mean(-1)

        # update and log metrics
        loss_metric = self.val_loss if prefix == "val" else self.test_loss
        loss_metric(loss)

        self.log(f"{prefix}/loss", loss_metric, on_step=True, on_epoch=True, prog_bar=True)

        to_log = {
            "data_0": batch,
            "gen_0": backwards_samples,
        }

        if self.nll_with_dem:
            forwards_samples = self.compute_and_log_nll(
                self.dem_cnf, self.prior, batch, prefix, "dem_"
            )
            to_log["gen_1_dem"] = forwards_samples
            self.compute_log_z(self.cfm_cnf, self.prior, backwards_samples, prefix, "dem_")
            self.compute_ess(self.cfm_cnf, self.prior, backwards_samples, prefix, "dem_")
        if self.nll_with_cfm:
            batch = self.energy_function.sample_test_set(self.eval_batch_size)
            batch = self.energy_function.normalize(batch)
            forwards_samples = self.compute_and_log_nll(
                self.cfm_cnf, self.cfm_prior, batch, prefix, ""
            )
            to_log["gen_1_cfm"] = forwards_samples

            iter_samples, _, _ = self.buffer.sample(self.eval_batch_size)

            # compute nll on buffer if not training cfm only
            if not self.hparams.debug_use_train_data and self.nll_on_buffer:
                forwards_samples = self.compute_and_log_nll(
                    self.cfm_cnf, self.cfm_prior, iter_samples, prefix, "buffer_"
                )

            if self.compute_nll_on_train_data:
                train_samples = self.energy_function.sample_train_set(self.eval_batch_size)
                train_samples = self.energy_function.normalize(train_samples)
                forwards_samples = self.compute_and_log_nll(
                    self.cfm_cnf, self.cfm_prior, train_samples, prefix, "train_"
                )

        if self.logz_with_cfm:
            #backwards_samples = self.cfm_cnf.generate(
            #    self.cfm_prior.sample(self.eval_batch_size),
            #)[-1]
            backwards_samples = self.energy_function.sample_test_set(self.eval_batch_size)
            backwards_samples = self.energy_function.normalize(backwards_samples)
            # backwards_samples = self.generate_cfm_samples(self.eval_batch_size)
            self.compute_log_z(self.cfm_cnf, self.cfm_prior, backwards_samples, prefix, "")
            self.compute_ess(self.cfm_cnf, self.cfm_prior, backwards_samples, prefix, "")
            

        self.eval_step_outputs.append(to_log)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("val", batch, batch_idx)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("test", batch, batch_idx)

    def eval_epoch_end(self, prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)
        # convert to dict of tensors assumes [batch, ...]
        outputs = {
            k: torch.cat([dic[k] for dic in self.eval_step_outputs], dim=0)
            for k in self.eval_step_outputs[0]
        }

        unprioritized_buffer_samples, cfm_samples = None, None
        if self.nll_with_cfm:
            unprioritized_buffer_samples, _, _ = self.buffer.sample(
                self.eval_batch_size,
                prioritize=self.prioritize_cfm_training_samples,
            )

            cfm_samples = self.cfm_cnf.generate(
                self.cfm_prior.sample(self.eval_batch_size),
            )[-1]
            #with torch.no_grad():
            #    cfm_samples =  self.cfm_cnf.reverse_fn#(self.cfm_prior.sample(self.eval_batch_size))[-1]
            cfm_samples = self.energy_function.unnormalize(cfm_samples)

            self.energy_function.log_on_epoch_end(
                self.last_samples,
                self.last_energies,
                wandb_logger,
                unprioritized_buffer_samples=unprioritized_buffer_samples,
                cfm_samples=cfm_samples,
                replay_buffer=self.buffer,
            )
            
            #log training data
            train_samples = self.energy_function.sample_train_set(self.eval_batch_size)
            
            self.energy_function.log_samples(
                train_samples,
                wandb_logger,
                name="train",
            )

        else:
            # Only plot dem samples
            self.energy_function.log_on_epoch_end(
                self.last_samples,
                self.last_energies,
                wandb_logger,
            )
        '''
        if "data_0" in outputs:
            # pad with time dimension 1
            names, dists = compute_distribution_distances(
                self.energy_function.unnormalize(outputs["gen_0"])[:, None],
                outputs["data_0"][:, None],
                self.energy_function,
            )
            names = [f"{prefix}/{name}" for name in names]
            d = dict(zip(names, dists))
            self.log_dict(d, sync_dist=True)
        '''
        self.eval_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self.eval_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        wandb_logger = get_wandb_logger(self.loggers)

        self.eval_epoch_end("test")
        self._log_energy_w2(prefix="test")
        self._log_data_w2(prefix="test")
        if self.energy_function.is_molecule:
           self._log_dist_w2(prefix="test")
           self._log_dist_total_var(prefix="test")
        else:
            self._log_data_total_var(prefix="test")

        batch_size = 1000
        final_samples = []
        n_batches = self.num_samples_to_save // batch_size
        print("Generating samples")
        for i in range(n_batches):
            start = time.time()
            samples = self.generate_samples(
                num_samples=batch_size,
                diffusion_scale=self.diffusion_scale,
                negative_time=self.hparams.negative_time,
            )
            final_samples.append(samples)
            end = time.time()
            print(f"batch {i} took {end - start:0.2f}s")

            if i == 0:
                self.energy_function.log_on_epoch_end(
                    samples,
                    self.energy_function(samples),
                    wandb_logger,
                )

        final_samples = torch.cat(final_samples, dim=0)
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        path = f"{output_dir}/samples_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path)
        print(f"Saving samples to {path}")
        import os

        os.makedirs(self.energy_function.name, exist_ok=True)
        path2 = f"{self.energy_function.name}/samples_{self.hparams.version}_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path2)
        print(f"Saving samples to {path2}")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        def _grad_fxn(t, x):
            return self.clipped_grad_fxn(
                t,
                x,
                self.energy_function,
                self.noise_schedule,
                self.num_estimator_mc_samples,
            )

        reverse_sde = VEReverseSDE(_grad_fxn, self.noise_schedule)

        self.prior = self.partial_prior(device=self.device, scale=self.noise_schedule.h(1) ** 0.5)
        if not self.energy_function._can_normalize:
            self.cfm_prior = self.partial_prior(device=self.device, scale=self.cfm_prior_std)
        else:
            self.cfm_prior = self.partial_prior(device=self.device, scale=self.noise_schedule.h(1) ** 0.5)
        if self.init_from_prior:
            init_states = self.prior.sample(self.num_init_samples)
        else:
            init_states = self.generate_samples(
                None, self.num_init_samples, diffusion_scale=self.diffusion_scale
            )
        init_energies = self.energy_function(init_states)
        
        self.energy_function.log_on_epoch_end(
                init_states, init_energies,
                get_wandb_logger(self.loggers)
            )
        
        self.buffer.add(init_states, init_energies.sum(-1))

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            self.cfm_net = torch.compile(self.cfm_net)


    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": self.hparams.lr_scheduler_update_frequency,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = DEMLitModule(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
