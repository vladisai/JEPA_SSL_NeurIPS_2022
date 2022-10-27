# Running experiments
Training script is run as follows:

```python train.py --configs <1 or more paths to .yaml files> --values wandb=True output_path=<path> model_type=VICReg vicreg.base_lr=42```

Important configuration options:
- `model_type`: defines which model to use, options are `VICReg`, `RSSM`, `SimCLR`.
- `dataset_noise`: controls changing noise level
- `dataset_static_noise`: controls fixed noise level
- `dataset_structured_noise`: controls whether the noise is structured.
- `output_path`: specifies the path to save models.
- `wandb`: enables weights and biases logging.
- `dataset_type`: chooses between one dot and three dot datasets. For one dot set `Single`, for three set `Multiple`.

To access particular model's options set options through the respective subconfig, e.g. `vicreg.base_lr`.

# Reproducing results
All configs are saved in `reproduce_configs` folder. To run a config from that folder, you can run

```python train.py --configs reproduce_configs/sweep_fixed_uniform.(1.25).vicreg.best.yaml```

This will run the best VICReg configuration for fixed uniform noise with coefficient 1.25.
