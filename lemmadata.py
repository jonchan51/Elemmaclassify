import functools
import collections

import parse
import datautils
import utils

@functools.lru_cache(maxsize=1)
def parse_problem(problemname):
    return parse.parse_cnf_file('E_conj/problems/{}'.format(problemname))

@utils.persist_to_file('lemmadata/usefulness.pickle')
def get_usefulness():
    print('getting usefulness')
    with open('E_conj/statistics', 'r') as f:
        s = f.read()
        ls = s.split('\n')
        usefulness = collections.defaultdict(dict)
        for l in ls:
            if not l.strip():
                continue
            psr, problem, lemmaname, *_ = l.split(':')
            psr = float(psr)
            lemmaname = lemmaname.split('.')[0]
            usefulness[problem][lemmaname] = psr

    return usefulness

@functools.lru_cache(maxsize=32)
def parse_problem(problemname):
    return parse.parse_cnf_file('E_conj/problems/{}'.format(problemname))

def _process_problemslemmas(l):
    name, lemma = l.split(':')
    _, problemname, lemmaname = name.split('/')
    return (
        problemname,
        lemmaname,
        parse_problem(problemname),
        parse.parse_cnf_clause(lemma),
        )

@utils.persist_to_file('lemmadata/problemslemmas.pickle')
def get_problemslemmas():
    print('parsing problems and lemmas')
    import multiprocessing

    with multiprocessing.Pool() as pool:
        with open('E_conj/lemmas') as f:
            return pool.map(_process_problemslemmas, f, 32)

def convert_to_datapoints(problemlemmas, usefulness):
    for pname, lname, problem, lemma in problemlemmas:
        yield ((problem, lemma), usefulness[pname][lname] < 1)

@functools.lru_cache(maxsize=1)
def get_dataset():
    usefulness = get_usefulness()
    train, crossval, test = datautils.split_by_amount(
            datautils.shuffle_by_hash(get_problemslemmas(), key=lambda x: str(x[:2])), [0.8, 0.9], lambda x: x)
    train_pos, train_neg = datautils.split_by_criteria(train, [lambda x: usefulness[x[0]][x[1]] < 1])
    crossval_pos, crossval_neg = datautils.split_by_criteria(crossval, [lambda x: usefulness[x[0]][x[1]] < 1])
    test_pos, test_neg = datautils.split_by_criteria(test, [lambda x: usefulness[x[0]][x[1]] < 1])

    # Shuffle order
    train_pos = datautils.shuffle_by_hash(train_pos, key=lambda x: str(x[:2]))
    train_neg = datautils.shuffle_by_hash(train_neg, key=lambda x: str(x[:2]))
    crossval_pos = datautils.shuffle_by_hash(crossval_pos, key=lambda x: str(x[:2]))
    crossval_neg = datautils.shuffle_by_hash(crossval_neg, key=lambda x: str(x[:2]))
    test_pos = datautils.shuffle_by_hash(test_pos, key=lambda x: str(x[:2]))
    test_neg = datautils.shuffle_by_hash(test_neg, key=lambda x: str(x[:2]))

    # Equalise classes
    num_examples = min(len(train_pos), len(train_neg))
    train_pos = train_pos[:num_examples]
    train_neg = train_neg[:num_examples]

    return (
            convert_to_datapoints(train_pos, usefulness),
            convert_to_datapoints(train_neg, usefulness),
            convert_to_datapoints(crossval_pos, usefulness),
            convert_to_datapoints(crossval_neg, usefulness),
            convert_to_datapoints(test_pos, usefulness),
            convert_to_datapoints(test_neg, usefulness),
            )

@utils.persist_iterator_to_file('lemmadata/dataset-train-pos.pickle')
def get_train_pos():
    yield from get_dataset()[0]

@utils.persist_iterator_to_file('lemmadata/dataset-train-neg.pickle')
def get_train_neg():
    yield from get_dataset()[1]

@utils.persist_iterator_to_file('lemmadata/dataset-crossval-pos.pickle')
def get_crossval_pos():
    yield from get_dataset()[2]

@utils.persist_iterator_to_file('lemmadata/dataset-crossval-neg.pickle')
def get_crossval_neg():
    yield from get_dataset()[3]

@utils.persist_iterator_to_file('lemmadata/dataset-test-pos.pickle')
def get_test_pos():
    yield from get_dataset()[4]

@utils.persist_iterator_to_file('lemmadata/dataset-test-neg.pickle')
def get_test_neg():
    yield from get_dataset()[5]

def get_train():
    # Interleave pos and neg examples
    for pair in zip(get_train_pos(), get_train_neg()):
        yield from pair

def get_crossval():
    yield from get_crossval_pos()
    yield from get_crossval_neg()

def get_test():
    yield from get_test_pos()
    yield from get_test_neg()

