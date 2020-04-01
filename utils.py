import sys
import pickle
import functools
import itertools
import subprocess

import numpy as np

def active_persist_to_file(filename):
    def decorator(original_func):
        cache = None

        @functools.wraps(original_func)
        def new_func(*params, filename=filename):
            nonlocal cache

            if cache is None:
                try:
                    cache = pickle.load(open(filename, 'rb'))
                except (IOError, ValueError):
                    cache = {}

            if params not in cache:
                cache[params] = original_func(*params)
                try:
                    pickle.dump(cache, open(filename, 'wb'))
                except:
                    sys.stderr.write("Couldn't save {}!".format(filename))
            return cache[params]

        return new_func

    return decorator

def persist_to_file(filename, key=lambda *params: params):
    def decorator(original_func):

        @functools.wraps(original_func)
        def new_func(*params, filename=filename):
            thisfilename = filename.format(*(tuple(sorted(p.items())) if isinstance(p, dict) else p for p in params))
            paramskey = key(*params)
            # print('Loading', paramskey, 'from', thisfilename)
            try:
                cache = pickle.load(open(thisfilename, 'rb'))
            except (IOError, ValueError):
                # print('File not found')
                cache = {}
                pickle.dump(cache, open(thisfilename, 'wb'))

            if paramskey not in cache:
                # print('File found, but key not found. Available keys:', list(cache.keys()))
                result = original_func(*params)
                cache = pickle.load(open(thisfilename, 'rb'))
                cache[paramskey] = result
                try:
                    pickle.dump(cache, open(thisfilename, 'wb'))
                except:
                    sys.stderr.write("Couldn't save {}!".format(filename.format(*params)))
            return cache[paramskey]

        return new_func

    return decorator

def persist_iterator_to_file(filename):
    def decorator(original_iterator):

        @functools.wraps(original_iterator)
        def new_func(*params, filename=filename):
            # print('Loading', params, 'from', filename.format(*params))
            try:
                with open(filename.format(*params), 'rb') as f:
                    # Check that the file contains the correct contents
                    pickled_params = pickle.load(f)
                    if params != pickled_params:
                        raise RuntimeError('Persisting iterators to file only works if every possible input is given a unique filename')

                    while True:
                        try:
                            yield pickle.load(f)
                        except EOFError:
                            break

            except (IOError, ValueError, EOFError):
                # Save the whole iterator first.
                with open(filename.format(*params), 'wb') as f:
                    # Save params as a check that the file is actually correct
                    pickle.dump(params, f)
                    for obj in original_iterator():
                        pickle.dump(obj, f)

                # Now return the actual iterator with recursive call
                yield from new_func(*params)

        return new_func

    return decorator

def persist_iterator_to_file_strparams(filename):
    def decorator(original_iterator):

        @functools.wraps(original_iterator)
        def new_func(*params, filename=filename):
            # print('Loading', params, 'from', filename.format(*params))
            strparams = str(params)
            try:
                with open(filename.format(*params), 'rb') as f:
                    # Check that the file contains the correct contents
                    pickled_params = pickle.load(f)
                    if strparams != pickled_params:
                        raise RuntimeError('Persisting iterators to file only works if every possible input is given a unique filename')

                    while True:
                        try:
                            yield pickle.load(f)
                        except EOFError:
                            break

            except (IOError, ValueError, EOFError):
                # Save the whole iterator first.
                with open(filename.format(*params), 'wb') as f:
                    # Save params as a check that the file is actually correct
                    pickle.dump(strparams, f)
                    for obj in original_iterator():
                        pickle.dump(obj, f)

                # Now return the actual iterator with recursive call
                yield from new_func(*params)

        return new_func

    return decorator

def randvec(shape):
    return np.random.normal(np.zeros(shape))

def normalize(vec):
    return vec / np.linalg.norm(vec)

def approx_int_square_root(n):
    """ Returns the largest divisor of n less than sqrt(n). """
    return next(i for i in range(int(n ** 0.5) + 1, 0, -1) if n % i == 0)

def twodfy(tensor):
    """ Try to make 1d tensors into a squareish shape """

    if len(tensor.shape) == 1:
        return tensor.reshape((approx_int_square_root(tensor.shape[0]), -1))
    return tensor

def group_into(it, group_size):
    # becomes zip(it, it, it, ...)
    # so each next call goes into each tuple position
    return zip(*(it,) * group_size)

def is_git_clean():
    status = subprocess.check_output(['git', 'status', '--porcelain'])
    return status == b''

def git_hash():
    githash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return githash.decode('utf-8').strip()
