import collections
import itertools
import datetime
import json
import queue

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.tensorboard as torchboard

import datautils
import utils

TrainSettings = collections.namedtuple(
        'TrainSettings',
        '''
        numepochs
        num_procs
        testingbatchsize
        loss_every_i_batches
        stats_every_i_batches
        weights_every_i_batches
        save_every_i_batches
        crossval_every_i_batches
        start_from_batch
        ''')

TrainSettings.__new__.__defaults__ = (
        -1,
        8,
        1024,
        1,
        5,
        5,
        50,
        50,
        0,
        )

def log_stats(writer, stats, global_counter):
    acc, precision, recall, fscore = datautils.calc_stats(stats)
    writer.add_scalar('Accuracy/train', acc, global_counter)
    writer.add_scalar('Precision/train', precision, global_counter)
    writer.add_scalar('Recall/train', recall, global_counter)
    writer.add_scalar('F-Score/train', fscore, global_counter)

def log_crossval(rank, world_size, model, loss_func, get_crossval, batch_queue, batchsize, global_counter):
    stats = datautils.Stats()

    with torch.no_grad():
        count = 0
        crossval_iter = get_crossval()
        firstbatch = list(itertools.islice(crossval_iter, world_size))
        count += world_size

        def is_nonempty(iterable):
            try:
                first = next(iterable)
            except StopIteration:
                return False, itertools.chain([]) # return empty iterable
            return True, itertools.chain([first], iterable)

        errloss = torch.tensor(0.)
        regloss = torch.tensor(0.)

        firstpoint = firstbatch[rank]

        nonempty, crossval_iter = is_nonempty(crossval_iter)

        if rank == 0:
            # Put the remaining points in the queue
            for point in crossval_iter:
                batch_queue.put(point)
                count += 1

        # If the queue is meant to be populated,
        # make sure everyone sees something in the queue before starting
        if nonempty:
            while batch_queue.empty() or batch_queue.qsize() == 0:
                continue

            dist.barrier()

        batchx = []
        batchy = []
        preds = []

        batchx.append(firstpoint[0])
        batchy.append(firstpoint[1])
        preds.append(model([firstpoint[0]]))

        while True:
            try:
                point = batch_queue.get(block=False)
                batchx.append(point[0])
                batchy.append(point[1])
                preds.append(model([point[0]]))
            except:
                if batch_queue.qsize() == 0 and batch_queue.empty():
                    break

        preds = torch.cat(preds)

        for pred, y in zip(preds, batchy):
            stats = datautils.update_stats(stats, bool(pred > 0), y)

        target = torch.tensor([1. if y else 0. for y in batchy])
        errloss, regloss = loss_func(preds, target)
        errloss *= len(batchx)
        dist.all_reduce(errloss)
        errloss /= count

        acc, precision, recall, fscore = datautils.calc_stats(stats)

        return errloss, regloss, acc, precision, recall, fscore

        writer.add_scalar('Err-Loss/crossval', errloss, global_counter)
        writer.add_scalar('Reg-Loss/crossval', regloss, global_counter)
        writer.add_scalar('Accuracy/crossval', acc, global_counter)
        writer.add_scalar('Precision/crossval', precision, global_counter)
        writer.add_scalar('Recall/crossval', recall, global_counter)
        writer.add_scalar('F-Score/crossval', fscore, global_counter)

def log_weights(writer, model, global_counter):
    for name, weight in model.named_parameters():
        writer.add_image('weight_{}'.format(name), utils.twodfy(weight), global_counter, dataformats='HW')

def log_experiment_info(save_dir, experiment_name, loss_func, git_hash, now, experimentsettings, trainsettings):
    info_filename = '{}/trainv1-{}-{}.info'.format(save_dir, git_hash, experiment_name)
    with open(info_filename, 'w') as f:
        json.dump({
            'start_time': str(now),
            'git_hash': str(git_hash),
            'experimentsettings': str(experimentsettings),
            'trainsettings': str(trainsettings),
            'train version': 'trainv1',
            }, f, indent=4, sort_keys=True)

    short_git_hash = git_hash[:7]
    lossfunc_filename = '{}/trainv1-{}-{}.lossfunc'.format(save_dir, short_git_hash, experiment_name)
    torch.save(loss_func, lossfunc_filename)

def save_model(model, opt, save_dir, short_git_hash, experiment_name, epoch, global_counter):
    model_filename = '{}/trainv1-{}-{}-{}.model'.format(save_dir, short_git_hash, experiment_name, global_counter)
    optim_filename = '{}/trainv1-{}-{}-{}.optim'.format(save_dir, short_git_hash, experiment_name, global_counter)
    modelstatedict_filename = model_filename + '.statedict'
    optimstatedict_filename = optim_filename + '.statedict'
    print('(e: {} i: {}) Saving model to {}'.format(epoch, global_counter, model_filename))

    torch.save(model.state_dict(), modelstatedict_filename)
    torch.save(opt.state_dict(), optimstatedict_filename)
    torch.save(model, model_filename)
    torch.save(opt, optim_filename)

def train(
        rank,
        world_size,
        save_dir,
        experiment_name,
        model,
        loss_func,
        opt,
        get_train,
        get_crossval,
        batch_queue,
        experimentsettings,
        trainsettings={},
        git_hash=None,
        timestring=None,
        ):
    """ The big train function with all the settings """

    dist.init_process_group('gloo', init_method='env://', world_size=world_size, rank=rank)

    torch.set_num_threads(1)

    if not utils.is_git_clean():
        raise RuntimeError('Ensure that all changes have been committed!')

    git_hash = utils.git_hash() if git_hash is None else git_hash
    short_git_hash = git_hash[:7]

    # Start time string
    now = datetime.datetime.now()
    timestring = now.strftime('%Y%m%d%a-%H%M%S') if timestring is None else timestring

    experiment_name = '{}-{}'.format(experiment_name, timestring)

    batchsize = experimentsettings.batchsize

    numepochs = trainsettings.numepochs
    testingbatchsize = trainsettings.testingbatchsize
    loss_every_i_batches = trainsettings.loss_every_i_batches
    stats_every_i_batches = trainsettings.stats_every_i_batches
    save_every_i_batches = trainsettings.save_every_i_batches
    weights_every_i_batches = trainsettings.weights_every_i_batches
    start_from_batch = trainsettings.start_from_batch

    # Log experiment information
    if start_from_batch == 0:
        log_experiment_info(save_dir, experiment_name, loss_func, git_hash, now, experimentsettings, trainsettings)

    if rank == 0:
        writer = torchboard.SummaryWriter(log_dir='runs/trainv1-{}-{}'.format(short_git_hash, experiment_name))

    try:
        epoch_iterator = range(numepochs) if numepochs is not -1 else itertools.count()
        dataset_iterator = ((epoch, point) for epoch in epoch_iterator for point in get_train())
        batch_iterator = utils.group_into(dataset_iterator, batchsize)
        stats = datautils.Stats()
        global_counter = batchsize * start_from_batch
        epoch = None

        for batchnum, batch__ in itertools.islice(enumerate(batch_iterator), start_from_batch, None):
            batch__ = list(batch__)
            cur_epoch = batch__[0][0]

            if rank == 0:
                print('(e: {} i: {}) Training batch #{}'.format(epoch, global_counter, batchnum))

            if rank == 0 and batchnum % save_every_i_batches == 0:
                save_model(model, opt, save_dir, short_git_hash, experiment_name, epoch, global_counter)

            if epoch is None:
                epoch = cur_epoch

            if cur_epoch > epoch:
                if rank == 0:
                    print('(e: {} i: {}) Testing cross-validation set'.format(epoch, global_counter))
                epoch = cur_epoch

                errloss, regloss, acc, precision, recall, fscore = log_crossval(rank, world_size, model, loss_func, get_crossval, batch_queue, testingbatchsize, global_counter)

                if rank == 0:
                    writer.add_scalar('Err-Loss/crossval', errloss, global_counter)
                    writer.add_scalar('Reg-Loss/crossval', regloss, global_counter)
                    writer.add_scalar('Accuracy/crossval', acc, global_counter)
                    writer.add_scalar('Precision/crossval', precision, global_counter)
                    writer.add_scalar('Recall/crossval', recall, global_counter)
                    writer.add_scalar('F-Score/crossval', fscore, global_counter)

            if rank == 0 and batchnum % stats_every_i_batches == 0:
                log_stats(writer, stats, global_counter)
                stats = datautils.Stats()

            if rank == 0 and batchnum % weights_every_i_batches == 0:
                log_weights(writer, model, global_counter)

            # batch__ = list(batch__)
            # batch_ = list(batch__[rank::world_size])
            # cur_epoch = [point[0] for point in batch__][0]
            # batch = [point[1] for point in batch_]

            dist.barrier()

            # Why do batching when most of the time is spent computing gradients?
            # The runtime of the forward direction is roughly linear in the number of created tensors,
            # thanks to caching
            # So by making each process spend roughly more equal amounts of time in the forward direction,
            # we indirectly make the # of created tensors more equal,
            # which indirectly makes the time spent computing gradients more equal.
            # The speed up isn't perfect, but it's still a speedup (3x speedup on 8 cores for relatively small batch sizes like 64)
            # Speed up will be better if the forward direction takes more time than the backward direction,
            # so speed up should be better on larger batch sizes.

            # Make sure that every process gets at least one point
            firstpoint = batch__[rank][1]

            if rank == 0:
                # Put the remaining points in the queue
                for point in batch__[world_size:]:
                    batch_queue.put(point[1])

            # If the queue is meant to be populated,
            # make sure everyone sees something in the queue before starting
            if len(batch__) > world_size:
                while batch_queue.empty() or batch_queue.qsize() == 0:
                    continue

                dist.barrier()

            # batchx = [point[0] for point in batch]
            # batchy = [point[1] for point in batch]
            # preds = parallelmodel(batchx)

            batchx = []
            batchy = []
            preds = []

            firstpred = model([firstpoint[0]])
            batchx.append(firstpoint[0])
            batchy.append(firstpoint[1])
            preds.append(firstpred)

            while True:
                try:
                    point = batch_queue.get(block=False)
                    pred = model([point[0]])
                    batchx.append(point[0])
                    batchy.append(point[1])
                    preds.append(pred)
                except queue.Empty:
                    if batch_queue.qsize() == 0 and batch_queue.empty():
                        break

            preds = torch.cat(preds)

            for pred, y in zip(preds, batchy):
                with torch.no_grad():
                    stats = datautils.update_stats(stats, bool(pred > 0), y)

            target = torch.tensor([1. if y else 0. for y in batchy])
            errloss, regloss = loss_func(preds, target)
            errloss.backward(retain_graph=True)

            for param in model.parameters():
                param.grad.data *= len(preds)
                dist.all_reduce(param.grad.data)
                param.grad.data /= len(batch__)

            regloss.backward()

            if rank == 0 and batchnum % loss_every_i_batches == 0:
                print('(e: {} i: {}) Loss: {:.5f}\t{:.5f}'.format(epoch, global_counter, errloss.item(), regloss.item()))
                writer.add_scalar('Err-Loss/train', errloss.item(), global_counter)
                writer.add_scalar('Reg-Loss/train', regloss.item(), global_counter)

            dist.barrier()

            if rank == 0:
                opt.step()
            opt.zero_grad()

            global_counter += len(batch__)

    except KeyboardInterrupt:
        pass
