import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from lightning import LightningModule
from functools import partial
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from .dem_module import *
from .components.energy_net_wrapper import EnergyNet
from .components.score_estimator import log_expectation_reward, estimate_grad_Rt
from .components.bootstrap_scheduler import BootstrapSchedule
from .components.ema import EMA
from fab.utils.plotting import plot_contours


class CBNEMLitModule(DEMLitModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        time_schedule: AnnealingSchedule,
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
        ais_steps: int = 0,
        ais_dt: float = 0.1,
        ais_warmup: int = 100,
        ema_beta=0.99,
        t0_regulizer_weight=0.,
        bootstrap_schedule: BootstrapSchedule = None,
        bootstrap_warmup: int = 2e3,
        bootstrap_mc_samples: int = 80,
        epsilon_train=1e-4,
        prioritize_warmup=0,
        iden_t=True,
        mh_iter=0,
        num_efficient_samples=0,
        bootstrap_from_checkpoint=True,
        tangent_normalization=False,
    ) -> None:
            
            net = partial(EnergyNet, net=net, 
                          noise_schedule=noise_schedule,
                          max_iter=mh_iter)
            super().__init__(
                net=net,
                optimizer=optimizer,
                scheduler=scheduler,
                energy_function=energy_function,
                noise_schedule=noise_schedule,
                lambda_weighter=lambda_weighter,
                buffer=buffer,
                num_init_samples=num_init_samples,
                num_estimator_mc_samples=num_estimator_mc_samples,
                num_samples_to_generate_per_epoch=num_samples_to_generate_per_epoch,
                num_samples_to_sample_from_buffer=num_samples_to_sample_from_buffer,
                num_samples_to_save=num_samples_to_save,
                eval_batch_size=eval_batch_size,
                num_integration_steps=num_integration_steps,
                lr_scheduler_update_frequency=lr_scheduler_update_frequency,
                nll_with_cfm=nll_with_cfm,
                nll_with_dem=nll_with_dem,
                nll_on_buffer=nll_on_buffer,
                logz_with_cfm=logz_with_cfm,
                cfm_sigma=cfm_sigma,
                cfm_prior_std=cfm_prior_std,
                use_otcfm=use_otcfm,
                nll_integration_method=nll_integration_method,
                use_richardsons=use_richardsons,
                compile=compile,
                prioritize_cfm_training_samples=prioritize_cfm_training_samples,
                input_scaling_factor=input_scaling_factor,
                output_scaling_factor=output_scaling_factor,
                clipper=clipper,
                score_scaler=score_scaler,
                partial_prior=partial_prior,
                clipper_gen=clipper_gen,
                diffusion_scale=diffusion_scale,
                cfm_loss_weight=cfm_loss_weight,
                use_ema=use_ema,
                ema_beta=ema_beta,
                use_exact_likelihood=use_exact_likelihood,
                debug_use_train_data=debug_use_train_data,
                init_from_prior=init_from_prior,
                compute_nll_on_train_data=compute_nll_on_train_data,
                use_buffer=use_buffer,
                tol=tol,
                version=version,
                negative_time=negative_time,
                num_negative_time_steps=num_negative_time_steps,
                ais_steps=ais_steps,
                ais_dt=ais_dt,
                ais_warmup=ais_warmup,
                iden_t=False,
                sample_noise=False
            )
            self.t0_regulizer_weight = t0_regulizer_weight
            self.bootstrap_scheduler = bootstrap_schedule
            self.epsilon_train = epsilon_train
            self.bootstrap_warmup = bootstrap_warmup
            self.bootstrap_mc_samples = bootstrap_mc_samples
            self.prioritize_warmup = prioritize_warmup
            assert self.num_estimator_mc_samples > self.bootstrap_mc_samples
            
            self.num_efficient_samples = num_efficient_samples
            self.reverse_sde = VEReverseSDE(self.net, 
                                                    self.noise_schedule, 
                                                    self.energy_function, None, num_efficient_samples)

            
            if use_ema:
                self.net = EMAWrapper(self.net)
            self.net.score_clipper = clipper_gen
            
            self.bootstrap_from_checkpoint = bootstrap_from_checkpoint
            self.time_schedule = time_schedule
            self.log_residual = False
            self.tangent_normalization = tangent_normalization
            
    def forward(self, t: torch.Tensor, x: torch.Tensor, with_grad=False) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(t, x, with_grad=with_grad)
    
    def energy_estimator(self, xt, t, num_samples, reduction=False):
        if self.ais_steps != 0 and self.iter_num < self.ais_warmup:
            return ais(xt, t, 
                       num_samples, self.ais_steps, 
                       self.noise_schedule, self.energy_function, 
                       dt=self.ais_dt, mode='energy', reduction=reduction)
        sigmas = self.noise_schedule.h(t).unsqueeze(1).sqrt()
        data_shape = list(xt.shape)[1:]
        noise = torch.randn(xt.shape[0], num_samples, *data_shape).to(xt.device)
        x0_t = noise * sigmas.unsqueeze(-1) + xt.unsqueeze(1)
        energy_est = self.energy_function(x0_t, smooth=True)
        if reduction:
            energy_est = torch.logsumexp(energy_est, dim=1) -\
                torch.log(torch.tensor(num_samples)).to(xt.device)
            return energy_est
        return energy_est
    
    def sum_energy_estimator(self, e, num_samples):
        return torch.logsumexp(e, dim=1) - torch.log(torch.tensor(num_samples)).to(e.device)
    

    def cbnem_loss(self, samples, times, num_hutchinson_samples=1):
        with torch.enable_grad():
            samples = samples.detach().requires_grad_(True)
            times = times.detach().requires_grad_(True)

            predicted_energy = self.net.forward_e(times, samples)

            time_derivative = torch.autograd.grad(
                predicted_energy, times, 
                grad_outputs=torch.ones_like(predicted_energy),
                retain_graph=True,
            )[0]

            grad_E = torch.autograd.grad(
                predicted_energy, samples, 
                grad_outputs=torch.ones_like(predicted_energy),
                create_graph=True, 
            )[0]

            grad_E_norm_squared = torch.norm(grad_E, dim=-1).pow(2)

            # Hutchinson's trick
            trace_hessian = torch.zeros(samples.shape[0], device=samples.device)
            for _ in range(num_hutchinson_samples):
                v = torch.randn_like(samples)  # Gaussian noise better than Rademacher for high-dim
                Hv = torch.autograd.grad(
                    grad_E, samples, v,
                    retain_graph=True, create_graph=False 
                )[0]
                trace_hessian += (v * Hv).sum(dim=-1)
            trace_hessian /= num_hutchinson_samples

            # PDE residual: - dE/dt + (sigma^2 / 2)*trace(Hessian(E)) 
            #               + (sigma^2 / 2)*||grad(E)||^2 = 0
            loss_terms = (
                time_derivative
                - 0.5 * self.noise_schedule.g(times) * trace_hessian
                - 0.5 * self.noise_schedule.g(times) * grad_E_norm_squared
            ).unsqueeze(-1).clone().detach()
            if self.tangent_normalization:
                loss_terms = loss_terms / (loss_terms.norm(dim=-1, keepdim=True) + 1e-6)
            loss = (2 * predicted_energy * loss_terms)

        self.log("train/loss", loss.mean(), on_step=False, on_epoch=True)
        # self.log("train/predicted_energy", predicted_energy.mean())
        self.log("train/time_derivative", time_derivative.mean())
        self.log("train/trace_hessian", trace_hessian.mean())
        self.log("train/grad_E_norm_squared", grad_E_norm_squared.mean())

        return loss

    def cbnem_loss_v2(self, samples, times):
        samples = samples.detach().requires_grad_(True)
        times = times.detach().requires_grad_(True)

        def energy_fn(samples, times):
            return -self.net.forward_e(times, samples).sum()
        
        def grad_x_fn(x):
            return torch.func.jacrev(energy_fn, argnums=0)(x, times)
        
        dE_dt = torch.func.jacrev(energy_fn, argnums=1)(samples, times)

        v = torch.randn_like(samples)
        dE_dx, Hv = torch.func.jvp(grad_x_fn, (samples,), (v,))
        trace_hessian = (v * Hv).sum(dim=-1)
        g_t = self.noise_schedule.g(times)
        dE_dx_norm2 = torch.norm(dE_dx.detach(), dim=-1).pow(2)
        r = (dE_dt - 0.5 * g_t * trace_hessian + 0.5 * g_t * dE_dx_norm2)
        r = r.unsqueeze(-1).detach()
        if self.tangent_normalization:
            r = r / (r.norm(dim=-1, keepdim=True) + 1e-6)
        E = energy_fn(samples, times).unsqueeze(-1)
        loss = 2 * E * r
        return loss
    
    def get_loss(self, times: torch.Tensor, 
                 samples: torch.Tensor, 
                 clean_samples: torch.Tensor,
                 train=False) -> torch.Tensor:
        
        if self.iter_num > 0:
            self.reverse_sde = VEReverseSDE(self.net, self.noise_schedule, 
                                            self.energy_function,None)
            

        continuous_loss = self.cbnem_loss(samples, times)
        # continuous_loss = self.cbnem_loss_v2(samples, times)
        
        # self.log(
        #         "energy_loss_t0",
        #         error_norms_t0.mean(),
        #         on_step=True,
        #         on_epoch=True,
        #         prog_bar=False,
        #     )
        # if not self.buffer.prioritize:
        #     self.log(
        #             "energy_loss",
        #             energy_error_norm.mean(),
        #             on_step=True,
        #             on_epoch=True,
        #             prog_bar=False,
        #         )
        # else:
        #     self.log(
        #             "energy_loss",
        #             torch.tensor(1e8).to(energy_error_norm.device),
        #             on_step=True,
        #             on_epoch=True,
        #             prog_bar=False,
        #         )
            
        self.iter_num += 1
        
        #if self.iter_num == self.prioritize_warmup:
        #    self.buffer.prioritize = False
        
        # self.log(
        #     "largest energy",
        #     energy_clean.min(),
        #         on_step=True,
        #         on_epoch=True,
        #         prog_bar=False,
        # )
        
        # self.log(
        #     "mean energy",
        #     energy_clean.mean(),
        #         on_step=True,
        #         on_epoch=True,
        #         prog_bar=False,
        # )

        full_loss = (continuous_loss).sum(-1)

        return full_loss
    
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
            metroplolis_hasting=(self.reverse_sde.mh_sample is not None)
        )
        if return_full_trajectory:
            return trajectory
        
        return trajectory[-1]
    
    def on_train_epoch_start(self):
        self.train_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        self.EMA.step_ema(self.ema_model, self.net)
        "Lightning hook that is called when a training epoch ends."
        self.log(
            "val/training_time",
            time.time() - self.train_start_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        if self.reverse_sde.mh_sample is not None:
            if self.clipper_gen is not None:
                self.ema_model.score_clipper = self.clipper_gen
                reverse_sde = VEReverseSDE(
                    self.clipper_gen.wrap_grad_fxn(self.ema_model), 
                    self.noise_schedule, self.energy_function,
                    self.ema_model.MH_sample,
                    num_efficient_samples = self.num_efficient_samples
                )
                
            else:
                reverse_sde = VEReverseSDE(
                    self.ema_model, self.noise_schedule, self.energy_function,
                    self.ema_model.MH_sample,
                    num_efficient_samples = self.num_efficient_samples
                )
        else:
            if self.clipper_gen is not None:
                reverse_sde = VEReverseSDE(
                    self.clipper_gen.wrap_grad_fxn(self.ema_model), 
                    self.noise_schedule, self.energy_function,
                    num_efficient_samples = self.num_efficient_samples
                )
                
            else:
                reverse_sde = VEReverseSDE(
                    self.ema_model, self.noise_schedule, self.energy_function,
                    num_efficient_samples = self.num_efficient_samples
                )
        sample_start_time = time.time()
        self.last_samples = self.generate_samples(
            reverse_sde=reverse_sde, diffusion_scale=self.diffusion_scale
        )
        self.log(
            "val/sampling_time",
            time.time() - sample_start_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
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
    
    def on_after_backward(self) -> None:
        
        if self.global_step % 1000 == 0:
            with torch.no_grad():
                total_norm = 0.0
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        param_norm = torch.norm(param.grad).item()
                        total_norm += param_norm ** 2
                        # self.log(f"gradients/{name}", param_norm, on_step=False, on_epoch=True)
                
                total_norm = total_norm ** 0.5  # Compute total gradient norm
                # self.log("gradients/total_norm", total_norm, on_step=False, on_epoch=True)

    def compute_fpe_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        sigma_t = self.noise_schedule.g(t)

        with torch.enable_grad():
            t = t.detach().requires_grad_(True)
            x = x.detach().requires_grad_(True)

            E_pred = self.net.forward_e(t, x)
            grad_E = torch.autograd.grad(E_pred, x, grad_outputs=torch.ones_like(E_pred), create_graph=True)[0]
            grad_norm_sq = torch.sum(grad_E**2, dim=-1)

            hessian_E = torch.autograd.grad(torch.sum(grad_E), x, create_graph=True)[0]
            trace_Hessian = torch.sum(hessian_E, dim=-1)

            time_derivative = torch.autograd.grad(E_pred, t, grad_outputs=torch.ones_like(E_pred), create_graph=True)[0]

        residual = time_derivative - 0.5*(sigma_t**2)*trace_Hessian + 0.5*(sigma_t**2)*grad_norm_sq

        return residual