#!/usr/bin/env python

import os,sys
import numpy as np
import time
import h5py

import argparse

def set_parser():
    parser = argparse.ArgumentParser(description='Toy_AL_sampling')
    
    parser.add_argument('--seed',           type=int,   default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--phase', type=int,            default=0,
                        help='the current phase idx, if non-zero, load ckpt')
    parser.add_argument('--num_train_sample', type=int, default=2000,
                        help='number of training samples')
    parser.add_argument('--num_test_sample',  type=int, default=500,
                        help='number of testing samples')
    parser.add_argument('--num_global_test_sample',  type=int, default=2000,
                        help='number of global testing samples')
    parser.add_argument('--sigma',  type=float, default=0.01,
                        help='sigma for Gaussian mixture sampling')
    parser.add_argument('--sample_input_filename', default='test_output.hdf5',
                        help='the filename of input data for sampling, including test output sample')
    parser.add_argument('--global_test_filename', default='global_test.hdf5',
                        help='filename of global test set used for all training')
    
    args = parser.parse_args()
    return args

def _get_test_output(fname):
    test_output = np.empty([0, 4])
    with h5py.File(fname, 'r') as f:
        test_output = f['final_test_output'][:]
    return test_output

def _save_training_input(val_train, sample_train, val_test, sample_test, fname):
    with h5py.File(fname, 'w') as f:
        f.create_dataset("X_train", data = val_train)
        f.create_dataset("y_train", data = sample_train)
        f.create_dataset("X_test", data = val_test)
        f.create_dataset("y_test", data = sample_test)

def _save_global_test(val_test, sample_test, fname):
    with h5py.File(fname, 'w') as f:
        f.create_dataset("X_global_test", data = val_test)
        f.create_dataset("y_global_test", data = sample_test)

def _sample_grid_uniform(num_sample, rng):
    return rng.uniform(0.0, 1.0, (num_sample, 3))

def _sample_grid_base_on_test(test_output, num_sample, sigma, rng):
    cov = sigma * sigma
    res = np.empty([0, 3])

    w = test_output[:,3]
    w = w / np.sum(w)
    sz = test_output.shape[0]
    freq = rng.multinomial(num_sample, w)

    for i in range(sz):
        temp_out = rng.multivariate_normal(np.array([test_output[i][0], test_output[i][1], test_output[i][2]]), np.diag(np.array([cov, cov, cov])), freq[i])
        temp_out = _remove_out_of_box(temp_out)
        res = np.concatenate([res, temp_out], axis=0)

    return res

def _remove_out_of_box(sample_in):
    assert sample_in.shape[1] == 3
    idx = (sample_in[:,0] <= 1.0) & (sample_in[:,1] <= 1.0) & (sample_in[:,2] <= 1.0) & \
          (sample_in[:,0] >= 0.0) & (sample_in[:,1] >= 0.0) & (sample_in[:,2] >= 0.0)
    sample_in = sample_in[idx]
    return sample_in

def _get_value(sample):
    x1 = sample[:,0]
    x2 = sample[:,1]
    x3 = sample[:,2]
    f1 = np.multiply(x1, np.cos(x2 + x3))
    f2 = np.multiply(x2, np.sin(np.multiply(x1, x3) + 4.0))
    f3 = np.multiply(np.cos(np.sqrt(np.square(x3) + np.square(x1))), np.square(np.log(np.square(x1) + np.power(x2, 4))))
    f4 = x1 - np.multiply(x2, np.exp(np.sin(x3) + np.cos(x1)))
    f5 = x3 + np.divide(np.multiply(x1, x2), (np.square(np.log(np.square(x2))) + np.square(np.sin(x1 + np.cos(x3))) + 1.0))
    h1 = f1 + np.divide(f2, f3 + 1.0)
    h2 = f2 + np.sqrt(np.square(f3) + np.square(f4) + np.square(f5))
    h3 = np.divide(f3, np.log(np.square(f1) + np.square(f4)) + 1.0)
    h4 = np.cos(f4) - np.divide(np.sqrt(np.square(f5) + np.square(f1)), 1.0 + np.square(f2))

    output = np.stack((h1, h2, h3, h4), axis=1)
    return output


def main():
    args = set_parser() 
    rng = np.random.default_rng(args.seed)

    if args.phase == 0:
        sample_global_test = _sample_grid_uniform(args.num_global_test_sample, rng)
        print("sample_global_test has shape {}".format(sample_global_test.shape))
        val_global_test = _get_value(sample_global_test)
        print("val_global_test has shape {}".format(val_global_test.shape))
        _save_global_test(val_global_test, sample_global_test, args.global_test_filename)

        sample_train = _sample_grid_uniform(args.num_train_sample, rng)
    else:
        test_output = _get_test_output(args.sample_input_filename) 
        print("test_output has shape {}".format(test_output.shape))
        sample_train = _sample_grid_base_on_test(test_output, args.num_train_sample, args.sigma, rng)

    print("sample_train has shape {}".format(sample_train.shape))
    val_train = _get_value(sample_train)
    print("val_train has shape {}".format(val_train.shape))

    sample_test = _sample_grid_uniform(args.num_test_sample, rng)
    print("sample_test has shape {}".format(sample_test.shape))
    val_test = _get_value(sample_test)
    print("val_test has shape {}".format(val_test.shape))

    _save_training_input(val_train, sample_train, val_test, sample_test, "training_input_phase_{}.hdf5".format(args.phase))

if __name__ == '__main__':
    main()

