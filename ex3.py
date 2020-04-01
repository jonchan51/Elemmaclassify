'''
Experiment 3 runner

Usage:
    ex3.py <experiment_id> [--train-settings=<trainsettings>] [--git_hash=<git_hash> --timestring=<timestring>]

Options:
    --train-settings=<trainsettings>  Comma separated list of key:value pairs.
    --git_hash=<git_hash>             If starting from saved model, the full git commit hash of the commit that ran this model.
    --timestring=<timestring>         If starting from saved model, the time string used when saving these models.
'''

import sys
import collections
import os
import random

import docopt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

import HRRTorch
import HRRClassifier
import trainv1
import lemmadata
import datautils

ExperimentSettings = collections.namedtuple(
        'ExperimentSettings',
        '''
        comment

        batchsize

        hrr_class
        hrr_size
        num_decoders
        num_classifier_layers
        num_classifier_hidden_neurons
        classifier_nonlinearity
        unitvecloss_weight
        sumvecloss_weight
        mlpweightloss_weight

        adam_lr
        adam_beta1
        adam_beta2
        adam_weight_decay
        ''')

model_experiments = [

        # Model experiments
        ExperimentSettings('Model-H16D0L2ReLU',      16, 'FlatTreeHRRTorch2', 16, 0,  2, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H16D0L3ReLU',      16, 'FlatTreeHRRTorch2', 16, 0,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H16D8L2ReLU',      16, 'FlatTreeHRRTorch2', 16, 8,  2, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H16D8L3ReLU',      16, 'FlatTreeHRRTorch2', 16, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H64D0L2ReLU',      16, 'FlatTreeHRRTorch2', 64, 0,  2, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H64D0L3ReLU',      16, 'FlatTreeHRRTorch2', 64, 0,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H64D8L2ReLU',      16, 'FlatTreeHRRTorch2', 64, 8,  2, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H64D8L3ReLU',      16, 'FlatTreeHRRTorch2', 64, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H64D16L2ReLU',     16, 'FlatTreeHRRTorch2', 64, 16, 2, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H64D16L3ReLU',     16, 'FlatTreeHRRTorch2', 64, 16, 3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H16D0L2Sigmoid',   16, 'FlatTreeHRRTorch2', 16, 0,  2, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H16D0L3Sigmoid',   16, 'FlatTreeHRRTorch2', 16, 0,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H16D8L2Sigmoid',   16, 'FlatTreeHRRTorch2', 16, 8,  2, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H16D8L3Sigmoid',   16, 'FlatTreeHRRTorch2', 16, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H16D0L2Sigmoid',   16, 'FlatTreeHRRTorch2', 64, 0,  2, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H64D0L3Sigmoid',   16, 'FlatTreeHRRTorch2', 64, 0,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H64D8L2Sigmoid',   16, 'FlatTreeHRRTorch2', 64, 8,  2, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H64D8L3Sigmoid',   16, 'FlatTreeHRRTorch2', 64, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H64D16L2Sigmoid',  16, 'FlatTreeHRRTorch2', 64, 16, 2, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),
        ExperimentSettings('Model-H64D16L3Sigmoid',  16, 'FlatTreeHRRTorch2', 64, 16, 3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-3, 0.9, 0.999, 0),

        ]

adam_experiments = [

        # Adam LR experiments
        ExperimentSettings('LR-3e-3-H16D8L3ReLU',    16, 'FlatTreeHRRTorch2', 16, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 3e-3, 0.9, 0.999, 0),
        ExperimentSettings('LR-3e-3-H64D8L3ReLU',    16, 'FlatTreeHRRTorch2', 64, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 3e-3, 0.9, 0.999, 0),
        ExperimentSettings('LR-3e-3-H16D8L3Sigmoid', 16, 'FlatTreeHRRTorch2', 16, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 3e-3, 0.9, 0.999, 0),
        ExperimentSettings('LR-3e-3-H64D8L3Sigmoid', 16, 'FlatTreeHRRTorch2', 64, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 3e-3, 0.9, 0.999, 0),
        ExperimentSettings('LR-1e-2-H16D8L3ReLU',    16, 'FlatTreeHRRTorch2', 16, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-2, 0.9, 0.999, 0),
        ExperimentSettings('LR-1e-2-H64D8L3ReLU',    16, 'FlatTreeHRRTorch2', 64, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-2, 0.9, 0.999, 0),
        ExperimentSettings('LR-1e-2-H16D8L3Sigmoid', 16, 'FlatTreeHRRTorch2', 16, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-2, 0.9, 0.999, 0),
        ExperimentSettings('LR-1e-2-H64D8L3Sigmoid', 16, 'FlatTreeHRRTorch2', 64, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-2, 0.9, 0.999, 0),
        ExperimentSettings('LR-3e-2-H16D8L3ReLU',    16, 'FlatTreeHRRTorch2', 16, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 3e-2, 0.9, 0.999, 0),
        ExperimentSettings('LR-3e-2-H64D8L3ReLU',    16, 'FlatTreeHRRTorch2', 64, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 3e-2, 0.9, 0.999, 0),
        ExperimentSettings('LR-3e-2-H16D8L3Sigmoid', 16, 'FlatTreeHRRTorch2', 16, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 3e-2, 0.9, 0.999, 0),
        ExperimentSettings('LR-3e-2-H64D8L3Sigmoid', 16, 'FlatTreeHRRTorch2', 64, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 3e-2, 0.9, 0.999, 0),
        ExperimentSettings('LR-1e-1-H16D8L3ReLU',    16, 'FlatTreeHRRTorch2', 16, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-1, 0.9, 0.999, 0),
        ExperimentSettings('LR-1e-1-H64D8L3ReLU',    16, 'FlatTreeHRRTorch2', 64, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 1e-1, 0.9, 0.999, 0),
        ExperimentSettings('LR-1e-1-H16D8L3Sigmoid', 16, 'FlatTreeHRRTorch2', 16, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-1, 0.9, 0.999, 0),
        ExperimentSettings('LR-1e-1-H64D8L3Sigmoid', 16, 'FlatTreeHRRTorch2', 64, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 1e-1, 0.9, 0.999, 0),
        ExperimentSettings('LR-3e-1-H16D8L3ReLU',    16, 'FlatTreeHRRTorch2', 16, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 3e-1, 0.9, 0.999, 0),
        ExperimentSettings('LR-3e-1-H64D8L3ReLU',    16, 'FlatTreeHRRTorch2', 64, 8,  3, 64, 'LeakyReLU', 0.1, 0.1, 0.005, 3e-1, 0.9, 0.999, 0),
        ExperimentSettings('LR-3e-1-H16D8L3Sigmoid', 16, 'FlatTreeHRRTorch2', 16, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 3e-1, 0.9, 0.999, 0),
        ExperimentSettings('LR-3e-1-H64D8L3Sigmoid', 16, 'FlatTreeHRRTorch2', 64, 8,  3, 64, 'Sigmoid',   0.1, 0.1, 0.005, 3e-1, 0.9, 0.999, 0),

        ]


round2_experiments = []
for batchsize in 16, 64, 256:
    for activation in 'LeakyReLU', 'Sigmoid':
        for hrr_size in 64, 128, 256, 512, 1024:
            for num_decoders in 16, 32, 64, 128:
                for num_classifier_layers in 2, 3:
                    for mlp_reg in 0.003, 0.001, 0.0003:
                        for LR in 1e-3, 3e-4, 1e-4:
                            for hrr_class in 'FlatTreeHRRTorch2', 'FlatTreeHRRTorchComp':
                                round2_experiments.append(
                                    ExperimentSettings(
                                        'Round2-{}-B{}H{}D{}L{}{}G{}LR{}'.format(
                                            hrr_class, batchsize, hrr_size, num_decoders, num_classifier_layers, activation, mlp_reg, LR),
                                        batchsize,
                                        hrr_class,
                                        hrr_size,
                                        num_decoders,
                                        num_classifier_layers,
                                        64,
                                        activation,
                                        0.1,
                                        0.1,
                                        mlp_reg,
                                        LR,
                                        0.9,
                                        0.999,
                                        0))

# Round 2 experiments (randomly keep 40 of them)
round2_experiments = datautils.shuffle_by_hash(round2_experiments)[:40]

# Starts from index 80

recnet_experiments = []
for batchsize in [16]:#, 64, 256:
    for activation in ['Sigmoid']:#'LeakyReLU', 'Sigmoid':
        for repr_size in 16, 64, 128, 256:
            for mlp_reg in 0.003, 0.001, 0.0003:
                for num_classifier_layers in 2, 3:
                    for LR in 1e-3, 3e-4, 1e-4:
                        for hrr_class in ['LSTreeM']:
                            recnet_experiments.append(
                                ExperimentSettings(
                                    'RecNet1-{}-B{}H{}L{}{}G{}LR{}'.format(
                                        hrr_class, batchsize, repr_size, num_classifier_layers, activation, mlp_reg, LR),
                                    batchsize,
                                    hrr_class,
                                    repr_size,
                                    'unused',
                                    num_classifier_layers,
                                    64,
                                    activation,
                                    0.1,
                                    0.1,
                                    mlp_reg,
                                    LR,
                                    0.9,
                                    0.999,
                                    0))

recnet_experiments = datautils.shuffle_by_hash(recnet_experiments)[:40]

experiments = model_experiments + adam_experiments + round2_experiments + recnet_experiments

def make_model(experimentsettings):
    hrr_size = experimentsettings.hrr_size
    num_decoders = experimentsettings.num_decoders
    num_classifier_layers = experimentsettings.num_classifier_layers
    num_classifier_hidden_neurons = experimentsettings.num_classifier_hidden_neurons
    classifier_nonlinearity = experimentsettings.classifier_nonlinearity
    if experimentsettings.hrr_class == 'FlatTreeHRRTorch2':
        hrrmodel = HRRTorch.FlatTreeHRRTorch2(hrr_size)
        featurizer = HRRClassifier.Decoder_featurizer(hrr_size, num_decoders)
    elif experimentsettings.hrr_class == 'FlatTreeHRRTorchComp':
        hrrmodel = HRRTorch.FlatTreeHRRTorchComp(hrr_size)
        featurizer = HRRClassifier.DecoderComp_featurizer(hrr_size, num_decoders)
    elif experimentsettings.hrr_class == 'LSTreeM':
        hrrmodel = HRRTorch.LSTreeM(hrr_size)
        featurizer = HRRClassifier.Cat_featurizer(hrr_size)
    nonlinearity = {
            'Sigmoid': nn.Sigmoid(),
            'LeakyReLU': nn.LeakyReLU(),
            }[classifier_nonlinearity]
    classifier = HRRClassifier.MLP_classifier(
            featurizer.get_output_size(),
            [num_classifier_hidden_neurons] * (num_classifier_layers - 1),
            nonlinearity,
            )

    return HRRClassifier.HRRClassifier(
            hrr_size,
            hrrmodel,
            featurizer,
            classifier,
            )

def make_loss(model, experimentsettings):
    return HRRClassifier.HRRClassifierLoss(model, experimentsettings)

def make_opt(model, experimentsettings):
    lr = experimentsettings.adam_lr
    beta1 = experimentsettings.adam_beta1
    beta2 = experimentsettings.adam_beta2
    weight_decay = experimentsettings.adam_weight_decay

    return optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

def main(rank, world_size, model, loss_func, opt, batch_queue, experiment_id, trainsettings, experimentsettings, git_hash=None, timestring=None):
    trainv1.train(
            rank,
            world_size,
            'ex3data',
            'ex3-{}-{}'.format(experiment_id, experimentsettings.comment),
            model,
            loss_func,
            opt,
            lemmadata.get_train,
            lemmadata.get_crossval,
            batch_queue,
            experimentsettings,
            trainsettings,
            git_hash,
            timestring,
            )

if __name__ == '__main__':
    settings = docopt.docopt(__doc__)

    experiment_id = int(settings['<experiment_id>'])
    experimentsettings = experiments[experiment_id]

    if settings['--train-settings'] is not None:
        trainsettings = {
                key.strip(): int(value.strip())
                for pair in settings['--train-settings'].split(',')
                for key, value, *_ in [pair.split(':')]
                }
    else:
        trainsettings = {}

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(random.randint(10000,20000))

    trainsettings = trainv1.TrainSettings()._replace(**trainsettings)

    if trainsettings.num_procs > experimentsettings.batchsize:
        print('num_procs too large. Setting num_procs to', experimentsettings.batchsize, 'to match batchsize')
        trainsettings = trainsettings._replace(num_procs=experimentsettings.batchsize)

    torch.manual_seed(42)

    if settings['--git_hash'] is not None:
        git_hash = settings['--git_hash']
        global_counter = trainsettings.start_from_batch * experimentsettings.batchsize
        timestring = settings['--timestring']

        short_git_hash = git_hash[:7]
        experiment_name = 'ex3-{}-{}-{}'.format(experiment_id, experimentsettings.comment, timestring)

        model_filename = '{}/trainv1-{}-{}-{}.model'.format('ex3data', short_git_hash, experiment_name, global_counter)
        optimstatedict_filename = '{}/trainv1-{}-{}-{}.optim.statedict'.format('ex3data', short_git_hash, experiment_name, global_counter)
        model = torch.load(model_filename)
        loss_func = make_loss(model, experimentsettings)
        opt = make_opt(model, experimentsettings)
        optstatedict = torch.load(optimstatedict_filename)
        opt.load_state_dict(optstatedict)

        mp.spawn(
                main,
                args=(
                    trainsettings.num_procs,
                    model, loss_func, opt,
                    mp.get_context('spawn').Queue(),
                    experiment_id,
                    trainsettings,
                    experimentsettings,
                    git_hash,
                    timestring,
                    ),
                nprocs=trainsettings.num_procs,
                )

    else:
        model = make_model(experimentsettings)
        loss_func = make_loss(model, experimentsettings)
        opt = make_opt(model, experimentsettings)

        mp.spawn(
                main,
                args=(
                    trainsettings.num_procs,
                    model, loss_func, opt,
                    mp.get_context('spawn').Queue(),
                    experiment_id,
                    trainsettings,
                    experimentsettings,
                    ),
                nprocs=trainsettings.num_procs,
                )

