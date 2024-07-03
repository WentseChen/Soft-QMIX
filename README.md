# Soft-QMIX

This repo is heavily based on the [pymarl](https://github.com/benellis3/pymarl2)

## Installation

please follow the installation guide in the original repo [pymarl](https://github.com/benellis3/pymarl2)

## Run

To run the code, you can use the following command:

```bash
bash ./qmix.sh
```

> You have to change `entity: "xxx"` in the `src/config/default.yaml` file to your own wandb entity.

you can change the hyperparameters in the `qmix.sh` file. For instance to run protoss5v5 map, you can use the following command (default is terran5v5):

```bash
CUDA_VISIBLE_DEVICES=0 python src/main.py --config=qmix --env-config=sc2_gen_protoss
```

To run 10v10 map, you should change the `sc2_gen_terran.yaml` file

```yaml
capability_config:
    n_units: 10
    n_enemies: 10
```

## Publication

If you find this repository useful, please cite our [paper]([https://arxiv.org/abs/2207.05631](https://arxiv.org/pdf/2406.13930)):
```
@article{chen2024soft,
  title={Soft-QMIX: Integrating Maximum Entropy For Monotonic Value Function Factorization},
  author={Chen, Wentse and Huang, Shiyu and Schneider, Jeff},
  journal={arXiv preprint arXiv:2406.13930},
  year={2024}
}
```




