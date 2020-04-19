'''
Experiment 4 runner

Usage:
    ex4.py <experiment_id> [--train-settings=<trainsettings>] [--git_hash=<git_hash> --timestring=<timestring>]

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
        recnetweightloss_weight

        adam_lr
        adam_beta1
        adam_beta2
        adam_weight_decay
        ''')

recnet_experiments = []
for batchsize in [16]:  # , 64, 256:
    for activation in ['Sigmoid']:  # 'LeakyReLU', 'Sigmoid':
        for repr_size in 16, 64, 128, 256:
            for mlp_reg in 0.003, 0.001, 0.0003:
                for recnet_loss in 0.003, 0.001, 0.0003:
                    for num_classifier_layers in 2, 3:
                        for LR in 1e-3, 3e-4, 1e-4:
                            for hrr_class in ['LSTreeM']:
                                recnet_experiments.append(
                                    ExperimentSettings(
                                        'RecNet1-{}-B{}H{}L{}{}G{}RL{}LR{}'.format(
                                            hrr_class, batchsize, repr_size, num_classifier_layers, activation, mlp_reg, recnet_loss, LR),
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
                                        recnet_loss,
                                        LR,
                                        0.9,
                                        0.999,
                                        0))

recnet_experiments = datautils.shuffle_by_hash(recnet_experiments)[:40]

experiments = recnet_experiments


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
        featurizer = HRRClassifier.DecoderComp_featurizer(
            hrr_size, num_decoders)
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
        'ex4data',
        'ex4-{}-{}'.format(experiment_id, experimentsettings.comment),
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
    os.environ['MASTER_PORT'] = str(random.randint(10000, 20000))

    trainsettings = trainv1.TrainSettings()._replace(**trainsettings)

    if trainsettings.num_procs > experimentsettings.batchsize:
        print('num_procs too large. Setting num_procs to',
              experimentsettings.batchsize, 'to match batchsize')
        trainsettings = trainsettings._replace(
            num_procs=experimentsettings.batchsize)

    torch.manual_seed(42)

    if settings['--git_hash'] is not None:
        git_hash = settings['--git_hash']
        global_counter = trainsettings.start_from_batch * experimentsettings.batchsize
        timestring = settings['--timestring']

        short_git_hash = git_hash[:7]
        experiment_name = 'ex4-{}-{}-{}'.format(
            experiment_id, experimentsettings.comment, timestring)

        model_filename = '{}/trainv1-{}-{}-{}.model'.format(
            'ex4data', short_git_hash, experiment_name, global_counter)
        optimstatedict_filename = '{}/trainv1-{}-{}-{}.optim.statedict'.format(
            'ex4data', short_git_hash, experiment_name, global_counter)
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
