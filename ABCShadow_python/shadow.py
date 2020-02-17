#!/usr/bin/env python3
# # coding: utf-8
#
# Copyright (c) 2019 quentinl-c.
#
# This file is part of ABCShadow 
# (see https://github.com/quentinl-c/ABCShadow_article_assets).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import argparse
import configparser
import json
import os
import time
from math import exp, sqrt
from pathlib import Path

import numpy as np

from abc_shadow.abc_impl import (abc_shadow, binom_sampler, metropolis_sampler,
                                 normal_sampler)
from abc_shadow.mh_impl import binom_ratio, mh_post_sampler, norm_ratio
from abc_shadow.model.binomial__2params_graph_model import \
    Binomial2ParamsGraphModel
from abc_shadow.model.binomial_graph_model import BinomialGraphModel
from abc_shadow.model.binomial_model import BinomialModel
from abc_shadow.model.ising_graph_model import IsingGraphModel
from abc_shadow.model.ising_graph_model_ng import IsingGraphModelNG
from abc_shadow.model.norm_model import NormModel

from collections.abc import Iterable

ALGOS = ['abc_shadow', 'metropolis_hastings']

MODELS = ['normal',
          'binomial',
          'binomial_2params_graph',
          'binomial_graph',
          'ising_graph',
          'ising_ng_graph']

CHUNK_SIZE = 1000


def retrieve_vector(entry):
    if entry is None:
        err = "This entry does not exist"
        raise KeyError(err)

    vec = list(map(np.float, entry.split(',')))
    return np.array(vec)


def main():
    parser = argparse.ArgumentParser(description="Shadow Launcher")
    parser.add_argument("algo", choices=ALGOS)
    parser.add_argument("model", choices=MODELS)

    parser.add_argument("-c", "--configfile", required=True)

    parser.add_argument('-o', '--output-dir', type=str, default='./')

    arguments = parser.parse_args()

    record = dict()

    if not os.path.isfile(arguments.configfile):
        parser.error("The file {} does not exist!".format(
            arguments.configfile))

    output_dir = Path(arguments.output_dir)
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    config = configparser.ConfigParser()
    config.read(arguments.configfile)

    if arguments.model not in config.sections():
        err = "Model is not described by the conf file"
        raise ValueError(err)

    config_model = config[arguments.model]

    theta_0 = retrieve_vector(config_model.get('theta0'))
    theta_perfect = retrieve_vector(config_model.get('thetaPerfect'))
    delta = retrieve_vector(config_model.get('delta'))
    n = config_model.getint('n')
    iters = config_model.getint('iters')
    size = config_model.getint('size')

    print('============= SUMMARY =============')
    print('theta_0: {}'.format(theta_0))
    print('theta_perfect: {}'.format(theta_perfect))
    print('delta: {}'.format(delta))
    print('n: {}'.format(n))
    print('iters: {}'.format(iters))
    print('size: {}'.format(size))

    record['algo'] = arguments.algo
    record['model'] = arguments.model
    record['theta0'] = theta_0.tolist()
    record['theta_perf'] = theta_perfect.tolist()
    record['iters'] = iters
    record['n'] = n
    record['delta'] = delta.tolist() if isinstance(
        delta, np.ndarray) else delta

    timestamp = str(time.time())

    filename = '-'.join([record['algo'], record['model'], timestamp])
    filename += '.json'

    output_file = output_dir / filename

    if 'seed' in config_model:
        print("🎲  Let's make this random world determinist")
        seed = config_model.getint('seed')
        np.random.seed(seed)
        print('seed {} is enabled'.format(seed))

    if arguments.algo == 'abc_shadow':

        print("🚀 🚀 🚀 🚀  ABC SHADOW 🚀 🚀 🚀 🚀 ")

        # Default sampler
        sampler = metropolis_sampler

        if arguments.model == 'normal':
            model = NormModel(*theta_perfect)
            sampler = normal_sampler

        elif arguments.model == 'binomial':
            model = BinomialModel(*theta_perfect)
            sampler = binom_sampler

        elif arguments.model == 'binomial_graph':
            model = BinomialGraphModel(*theta_perfect)

        elif arguments.model == 'binomial_2params_graph':
            model = Binomial2ParamsGraphModel(*theta_perfect)

        elif arguments.model == 'ising_graph':
            model = IsingGraphModel(*theta_perfect)
            ext_size = config_model.getint('in_size')
            size = (size, ext_size)
            print('size has bee updated: {}'.format(size))
        elif arguments.model == 'ising_ng_graph':
            model = IsingGraphModelNG(*theta_perfect)
            ext_size = config_model.getint('in_size')
            size = (size, ext_size)
            print('size has bee updated: {}'.format(size))
        else:
            err = "Unknown model: {}".format(arguments.model)
            raise ValueError(err)

        print("📊 Model {} has been instanciated".format(arguments.model))

        sampler_it = config_model.getint('samplerIt')
        print("Sampler iterations: {}".format(sampler_it))

        if not isinstance(size, Iterable):
            size = [size]

        sim_data = config_model.getboolean('simData')

        if sim_data:
            print("Observation is being generated ...")
            y_obs = sampler(model, *size, 5000)
        else:
            y_obs = retrieve_vector(config_model.get('obs'))

        print("Data observed: {}".format(y_obs))
        record['y_obs'] = y_obs.tolist() if isinstance(
            y_obs, np.ndarray) else y_obs
        if 'mask' in config_model:
            mask = retrieve_vector(config_model.get('mask'))
            print(f"🎭  Mask has been set: {mask}")
        else:
            mask = None

        model.set_params(*theta_0)

        sampler_kwargs = {'it': sampler_it}
        if iters % CHUNK_SIZE == 0:
            chunk = int(iters / CHUNK_SIZE)
            iters = CHUNK_SIZE
            print(f"Execution is splitted into {chunk} chunks of {iters} iterations")
        else:
            chunk = 1
            print(f"No split, {iters} iterations wil be executed")
        posteriors = list()
        posteriors.append(theta_0)

        start_time = time.time()
        for c in range(chunk):
            samples = abc_shadow(model,
                                 theta_0,
                                 y_obs,
                                 delta,
                                 n,
                                 size,
                                 iters,
                                 sampler=sampler,
                                 sampler_kwargs=sampler_kwargs,
                                 mask=mask)
            posteriors.extend(samples)
            record['posteriors'] = [post.tolist() if isinstance(post, np.ndarray)
                                    else post for post in posteriors]

            with open(output_file, 'w') as o_file:
                json.dump(record, o_file)

            print(f"💾  End of chunk {c} records are savec in {output_file}")
            # Updates the input parameters
            print(theta_0)
            theta_0 = samples[-1]
            print(theta_0)
            time.sleep(1)

        end_time = time.time()

        print("DURANTION : {}".format(end_time - start_time))

    elif arguments.algo == 'metropolis_hastings':
        print("🚂 🚂 🚂 Metropolis Hasting Sampling 🚂 🚂 🚂 ")

        mask = config_model.get('mask')

        if mask is not None:
            print(f"Mask has been set: {mask}")

        if arguments.model == 'normal':
            ratio = norm_ratio
            y_obs = np.random.normal(
                theta_perfect[0], sqrt(theta_perfect[1]), size)

        elif arguments.model == 'binomial':
            ratio = binom_ratio
            n_p = theta_perfect[0]
            p_perfect = exp(theta_perfect[-1]) / (1 + exp(theta_perfect[-1]))
            p_0 = exp(theta_0[0]) / (1 + exp(theta_0[0]))

            theta_perfect = np.array([n_p, p_perfect])
            theta_0 = np.array([n_p, p_0])

            print("WARNING theta_perfect and theta_0 have been upadated")
            print('theta_0: {}'.format(theta_0))
            print('theta_perfect: {}'.format(theta_perfect))
            sim_data = config_model.getboolean('simData')
            if sim_data:
                print("Observation is being generated ...")
                y_obs = np.random.binomial(
                    theta_perfect[0], theta_perfect[1], size)
            else:
                y_obs = retrieve_vector(config_model.get('obs'))
            mask = [1, 0]

            print("mask is forced to: {}".format(mask))
        else:
            err = "metropolis hasting cannot be used on a graph model"
            raise ValueError(err)

        print(f"📊 Model {arguments.model} has been instanciated")

        print(f"Data observed: {y_obs}")

        posteriors = mh_post_sampler(
            theta_0, y_obs, delta, n, iters, ratio, mask=mask)
    else:
        err = "Given estimation algorithm {arguments.algo} not known"
        raise ValueError(err)

    print("💾  Save Record ... ")

    record['algo'] = arguments.algo
    record['model'] = arguments.model
    record['theta0'] = theta_0.tolist()
    record['theta_perf'] = theta_perfect.tolist()
    record['iters'] = iters
    record['n'] = n
    record['delta'] = delta.tolist() if isinstance(
        delta, np.ndarray) else delta
    record['y_obs'] = y_obs.tolist() if isinstance(
        y_obs, np.ndarray) else y_obs
    record['posteriors'] = [post.tolist() if isinstance(post, np.ndarray)
                            else post for post in posteriors]
    timestamp = str(time.time())

    filename = '-'.join([record['algo'], record['model'], timestamp])
    filename += '.json'

    output_file = output_dir / filename
    with open(output_file, 'w') as o_file:
        json.dump(record, o_file)

    print(f"📝 Record saved in {output_file}")


if __name__ == "__main__":
    main()
