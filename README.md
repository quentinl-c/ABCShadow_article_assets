# ABCShadow_article_assets
> Source code - notebooks and instructions to reproduce the experimentations presented in: Morpho-statistical description of networks through graph modelling and Bayesian inference.

### Setup

All experimentations were executed inside a Docker container (image `debian:latest`). To ensure the persistence, source code and outputs were stored in a Docker volume.

```bash
docker run -it -v your_volume:/data debian:latest /bin/bash
```

Required dependencies : `g++`, `make` and `python3`.

Compile the soucre code:
```bash
cd ABCShadow_cpp
make
```

### Run ABC Shadow on the real data

Sufficient statistics and size of the different teams are encoded in the file `journal_paper.json`. The parameters used to run the ABC Shadow are described in the configuration file `config_loria.txt`. 

The `bootstrap.py` script run parallel  executions of  the ABC Shadow (one for each team).


```bash
python3 bootstrap.py abc_estim journal_paper.json configs/config_loria.txt output_dir
```

### Posterior analysis

All the resuslts demonstrated in the arcticle are computed using the Jupyter notebooks in the `journal` directory.

#### Binomial + Simulated data

All results related to Section 4 "ABC shadow in practice : illustration on synthetic data" were computed in : `setup.ipynd`.

#### Application

All results related to Section 5 "Application" were computed in : `application.ipynd`

### Posterior simulation

Posterior simulations are needeed to compute the error metrics (the asymptotic std & the MC std).

Once the MAPs are computed (using `application.ipynd` e.g.) and encoded in a JSON file (in `maps_journal_paper.json` e.g.), you can run Gibbs simulation using the `bootstrap.py` script. As the estimation procedure, the different simulations are executed in parallel. 

```
python3 bootstrap.py gibbs_sim journal_paper.json maps_journal_paper.json configs/config_sim_map.txt output_dir
```

### Python implemention

A Python implementation is provided as well and was mainly used to produce the preliminary results related to the Binomial distribution.

#### Setup
To install the python dependencies:

```sh
cd ABCShadow_python
pip install -r requirements.txt
```

For efficiency reasons, a part of the code is written in Cython. Hence, it requires to be compiled :
```sh
python3 setup.py build_ext --inplace
```

#### Sampling from a Binomial distribution

Run ABC Shadow :

```
./shadow.py abc_shadow binomial -c config/binomial_config_abc.ini           
```

Run Metropolis-Hastings:

```
./shadow.py metropolis_hastings binomial -c config/binomial_config_mh.ini
```