This repository contains code used for the IIB Project - Better Diffusion Models for Molecular Modelling by Sunaabh Trivedi. 
It is forked from the nem repo used to implement BNEM (https://arxiv.org/abs/2409.09787), and updated to implement:
1. Annealed Bootstrapped Schedule
2. Continuous-Time BNEM (CBNEM)

<h1> Installation </h1>

```
# create micromamba environment
conda env create -f environment.yaml
conda activate dem
```

```
# install Python dependencies
pip install -r requirements.txt
```

<h1> Running Experiments </h1>
To run an experiment, choose a model and experiment configuration. For example:

```
# Run BNEM on GMM
python dem/train.py experiment=gmm_idem model=endem_bootstrap
```

To run BNEM with the <b>annealed bootstrap schedule</b>:

```
python dem/train.py experiment=gmm_idem model=endem_bootstrap \
    model.components.time_schedule="annealing" \
    model.components.bootstrap_schedule="annealing"
```

To run the <b>CBNEM</b> method:

```
python dem/train.py experiment=gmm_cbnem model=cbnem \
    model.optimizer.lr=0.00001 \
    trainer.gradient_clip_val=0.1 \
    model.use_ema=true
```

<h1> New Features </h1>

<h2>Annealed Bootstrap Schedule </h2>

Implements a noise-aware bootstrapping strategy that prioritides learning low-noise samples early in training, improving energy learning and reducing error propagation.

Key additions:

- `models/components/annealing_schedule.py`: defines AnnealingSchedule
- `models/components/bootstrap_scheduler.py`: defines AnnealingBootstrapSchedule

Usage, run any `endem_bootstrap` experiment and add the following flags:
```
model.components.time_schedule: "annealing"
model.components.bootstrap_schedule: "annealing"
```

or change these directly in the relevant configs/model/ entry.

<h2> Continuous-Time BNEM (CBNEM) </h2>
A continuous-time generalisation of BNEM, bootstrapping from an infinitesimally smaller previous time-step; the model is effectively trained to fit the log-density Fokker–Planck equation, avoiding discrete bootstrapping steps.

Key additions:
- `cbnem_module.py`: new PyTorch Lightning module for CBNEM
- `configs/model/cbnem.yaml`: model config for CBNEM
- `configs/experiment/gmm_cbnem.yaml`: GMM experiment config for CBNEM

Usage: 
```
python dem/train.py experiment=gmm_cbnem model=cbnem
```

<h1> Project Structure (Key Changes) </h1>

```
dem/
├── cbnem_module.py     # CBNEM implementation
├── models/
│   └── components/
│       ├── annealing_schedule.py       # AnnealingSchedule
│       └── bootstrap_scheduler.py      # AnnealingBootstrapSchedule
configs/
├── model/
│   └── cbnem.yaml      # CBNEM model config
├── experiment/
│   └── gmm_cbnem.yaml      # CBNEM GMM experiment config
```