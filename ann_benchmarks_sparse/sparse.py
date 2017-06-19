import sys
import heapq as hq
import gzip
import numpy
import time
import os
import multiprocessing
import multiprocessing.pool
import argparse
import pickle
import resource
import random
import math
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve # Python 3
import sklearn.preprocessing
from scipy.sparse import csr_matrix


# Set resource limits to prevent memory bombs
memory_limit = 12 * 2**30
soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
if soft == resource.RLIM_INFINITY or soft >= memory_limit:
    print('resetting memory limit from {} to {}'.format(soft, memory_limit))
    resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, hard))


class BaseANN(object):
    def use_threads(self):
        return True

    def use_csr_matrix(self):
        return False

    def use_hash(self):
        return False


class KnnResult(object):
    def __init__(self, k):
        self.k = k
        self.h = []

    def add(self, dist, obj):
        val = (-dist, obj)
        if len(self.h) < self.k:
            hq.heappush(self.h, val)
        elif self.h[0] < val:
            hq.heapreplace(self.h, val)

    def get(self):
        res = []
        for i in range(0, min(len(self.h), self.k)):
            val = hq.heappop(self.h)
            #res.append((val[1], -val[0]))
            res.append(val[1])
        return res[::-1]


class BruteForceSparse(BaseANN):
    def __init__(self):
        self.X = []
        self.lengths = []
        self.name = 'BruteForceSparse()'

    def fit(self, X):
        self.X = X
        self.lengths = []
        for v in X:
            self.lengths.append(math.sqrt(sum([x[1] * x[1] for x in v])))

    def query(self, q, k):
        qlen = math.sqrt(sum([x[1] * x[1] for x in q]))
        knn = KnnResult(k)
        done = object()
        for i, (v, vlen) in enumerate(zip(self.X, self.lengths)):
            x = iter(v)
            y = iter(q)
            val1 = next(x, done)
            val2 = next(y, done)
            dist = 0
            while val1 is not done and val2 is not done:
                if val1[0] == val2[0]:
                    dist += val1[1] * val2[1]
                    val1 = next(x, done)
                    val2 = next(y, done)
                elif val1[0] < val2[0]:
                    val1 = next(x, done)
                else:
                    val2 = next(y, done)
            knn.add(1.0 - dist/(qlen * vlen), i)
        return knn.get()


class NmslibSparse(BaseANN):
    def __init__(self, space, method_name, method_param, query_param):
        assert space == 'cosinesimil_sparse' or space == 'cosinesimil_sparse_fast'
        self._space = space
        self._method_name = method_name
        self._method_param = method_param
        self._query_param = query_param
        self.name = 'Nmslib(method_name={}, method_param={}, query_param={})'.format(method_name, method_param, query_param)

    def fit(self, X):
        import nmslib
        self._index = nmslib.init(method=self._method_name,
                                  space=self._space,
                                  data_type=nmslib.DataType.SPARSE_VECTOR)
        for i, x in enumerate(X):
            self._index.addDataPoint(i, x)
        self._index.createIndex(self._method_param)
        self._index.setQueryTimeParams(self._query_param)

    def query(self, v, n):
        ids, distances = self._index.knnQuery(v, n)
        return ids

    def freeIndex(self):
        #self._index.freeIndex()
        pass


class PySparnn(BaseANN):
    def __init__(self, num_indexes):
        self._num_indexes = num_indexes
        self.name = 'pysparnn(num_indexes={})'.format(num_indexes)
        self._index = None

    def fit(self, X):
        import pysparnn.cluster_index as ci
        ids = range(X.shape[0])
        self._index = ci.MultiClusterIndex(X, ids, num_indexes=self._num_indexes)

    def query(self, v, n):
        assert v.shape[0] == 1
        ids = self._index.search(v, return_distance=False, k=n)
        return [int(i) for i in ids[0]]

    def freeIndex(self):
        self._index = None

    def use_csr_matrix(self):
        return True


class AnnoySparseHash(BaseANN):
    def __init__(self, n_trees, n_features, search_k):
        from sklearn.feature_extraction import FeatureHasher
        self._n_trees = n_trees
        self._n_features = n_features
        self._search_k = search_k
        self._h = FeatureHasher(n_features=n_features)
        self.name = 'AnnoySparseHash(n_trees={}, n_features={}, search_k={})'.format(n_trees, n_features, search_k)
        self._index = None

    def fit(self, X):
        import annoy
        h_x = self._h.transform(X).toarray()
        self._index = annoy.AnnoyIndex(f=self._n_features, metric='angular')
        for i, x in enumerate(h_x):
            self._index.add_item(i, x.tolist())
        self._index.build(self._n_trees)

    def query(self, v, n):
        assert len(v) == 1
        h_v = self._h.transform(v).toarray()[0]
        return self._index.get_nns_by_vector(h_v.tolist(), n, self._search_k)

    def freeIndex(self):
        pass

    def use_hash(self):
        return True


def parse(idx_val):
    parts = idx_val.split()
    if len(parts) != 2:
        parts = idx_val.split(':')
    return [int(parts[0]), float(parts[1])]


def to_csr_matrix(z, dim):
    row = []
    col = []
    data = []
    for i, x in enumerate(z):
        for idx, val in x:
            row.append(i)
            col.append(idx)
            data.append(val)
    return csr_matrix((data, (row, col)), shape=(len(z), dim + 1))


def to_hash(z):
    data = []
    for x in z:
        item = {}
        for idx, val in x:
            item[str(idx)] = float(val)
        data.append(item)
    return data


def get_dataset(which, limit=-1, random_state=3, test_size=10000):
    cache = 'queries/%s-%d-%d-%d.pkl' % (which, test_size, limit, random_state)

    if os.path.exists(cache):
        with open(cache, 'rb') as fd:
            h = pickle.load(fd)
            X_train = h['X_train']
            X_test = h['X_test']
            X_train_csr = h['X_train_csr']
            X_test_csr = h['X_test_csr']
            X_train_h = h['X_train_h']
            X_test_h = h['X_test_h']
            return X_train, X_test, X_train_csr, X_test_csr, X_train_h, X_test_h

    local_fn = os.path.join('install', which)
    if os.path.exists(local_fn + '.gz'):
        f = gzip.open(local_fn + '.gz')
    else:
        f = open(local_fn + '.txt')

    X = []
    max_idx = 0
    for i, line in enumerate(f):
        v = [parse(x) for x in line.split()]
        idx = max([x[0] for x in v])
        if idx > max_idx:
            max_idx = idx
        X.append(v)
        if limit != -1 and len(X) == limit:
            break

    import sklearn.cross_validation

    X_train, X_test = sklearn.cross_validation.train_test_split(
        X, test_size=test_size, random_state=random_state)

    X_train_csr = to_csr_matrix(X_train, max_idx + 1)
    X_test_csr = [to_csr_matrix([x], max_idx + 1) for x in X_test]

    X_train_h = to_hash(X_train)
    X_test_h = [to_hash([x]) for x in X_test]

    with open(cache, 'wb') as fd:
        h = {
            'X_train': X_train,
            'X_test': X_test,
            'X_train_csr': X_train_csr,
            'X_test_csr': X_test_csr,
            'X_train_h': X_train_h,
            'X_test_h': X_test_h
        }
        pickle.dump(h, fd)

    return X_train, X_test, X_train_csr, X_test_csr, X_train_h, X_test_h


def run_algo(args, library, algo, queries, results_fn):
    pool = multiprocessing.Pool()
    X_train, _, X_train_csr, _, X_train_h, _ = pool.apply(get_dataset, [args.dataset, args.limit])
    pool.close()
    pool.join()

    t0 = time.time()
    if algo != 'bf':
        if algo.use_csr_matrix():
            algo.fit(X_train_csr)
        elif algo.use_hash():
            algo.fit(X_train_h)
        else:
            algo.fit(X_train)
    build_time = time.time() - t0
    print('Built index in {}'.format(build_time))

    best_search_time = float('inf')
    best_precision = 0.0  # should be deterministic but paranoid
    for i in range(3):    # Do multiple times to warm up page cache, use fastest
        t0 = time.time()

        def single_query(t):
            v, v_csr, v_h, correct = t
            if algo.use_csr_matrix():
                found = algo.query(v_csr, 10)
            elif algo.use_hash():
                found = algo.query(v_h, 10)
            else:
                found = algo.query(v, 10)
            return len(set(found).intersection(correct))

        if algo.use_threads():
            pool = multiprocessing.pool.ThreadPool()
            results = pool.map(single_query, queries)
        else:
            results = map(single_query, queries)

        k = float(sum(results))
        search_time = (time.time() - t0) / len(queries)
        precision = k / (len(queries) * 10)
        best_search_time = min(best_search_time, search_time)
        best_precision = max(best_precision, precision)

    output = [library, algo.name, build_time, best_search_time, best_precision]
    print(output)

    with open(results_fn, 'a') as fd:
        fd.write('\t'.join(map(str, output)) + '\n')


def get_queries(args):
    print('computing queries with correct results...')

    X_train, X_test, X_train_csr, X_test_csr, X_train_h, X_test_h = \
        get_dataset(which=args.dataset, limit=args.limit)

    bf = BruteForceSparse()
    # Prepare queries
    bf.fit(X_train)
    queries = []
    for x, x_csr, x_h in zip(X_test, X_test_csr, X_test_h):
        correct = bf.query(x, 10)
        queries.append((x, x_csr, x_h, correct))
        if len(queries) % 100 == 0:
            print(len(queries), '...')

    return queries


def get_algos(pivot_file):
    algos = {
        'bruteforce': [BruteForceSparse()],
        'Annoy': [],
        'sw-graph(nmslib)': [],
        'hnsw(nmslib)': [],
        'napp(nmslib)': [],
        'falconn': [],
        'pysparnn': [],
    }

    if 'Annoy' in algos:
        for n_trees in [100, 200, 400]:
            for n_features in [5, 10, 20, 40, 80]:
                for search_k in [#100, 200, 400,
                                 #1000, 2000, 4000,
                                 10000, 20000, 40000,
                                 #100000, 200000, 400000
                                 ]:
                    algos['Annoy'].append(AnnoySparseHash(n_trees, n_features, search_k))

    if 'sw-graph(nmslib)' in algos:
        for nn in [17, 30]:
            for ef in [10, 15, 20, 25]:
                for efconstr in [200, 800]:
                    algos['sw-graph(nmslib)'].append(
                        NmslibSparse('cosinesimil_sparse_fast',
                                     'small_world_rand',
                                     dict(NN=nn, efConstruction=efconstr),
                                     dict(efSearch=ef)))

    if 'hnsw(nmslib)' in algos:
        for m in [10, 30]:
            for ef in [10, 15, 20, 25]:
                for efconstr in [200, 800]:
                    algos['hnsw(nmslib)'].append(
                        NmslibSparse('cosinesimil_sparse_fast',
                                     'hnsw',
                                     dict(M=m, efConstruction=efconstr),
                                     dict(efSearch=ef)))

    if 'napp(nmslib)' in algos:
        for p in [1000, 4000]:
            for pi in [100, 200]:
                for ps in [5, 10, 15]:
                    algos['napp(nmslib)'].append(
                        NmslibSparse('cosinesimil_sparse_fast',
                                     'napp',
                                     dict(numPivot=p, numPivotIndex=pi, pivotFile=pivot_file),
                                     dict(numPivotSearch=ps)))

    if 'falconn' in algos:
        for h in [100, 400]:
            for b in [13, 20]:
                for d in [128]:
                    for p in [100]:
                        algos['falconn'].append(
                            NmslibSparse('cosinesimil_sparse_fast',
                                         'falconn',
                                         dict(num_hash_tables=h, num_hash_bits=b, feature_hashing_dimension=d),
                                         dict(num_probes=p)))

    if 'pysparnn' in algos:
        for num_indexes in [1, 4, 8, 16, 32, 64, 128]:
            algos['pysparnn'].append(PySparnn(num_indexes))

    return algos


def get_fn(base, args):
    fn = os.path.join(base, args.dataset)

    if args.limit != -1:
        fn += '-%d' % args.limit
    if os.path.exists(fn + '.gz'):
        fn += '.gz'
    else:
        fn += '.txt'

    d = os.path.dirname(fn)
    if not os.path.exists(d):
        os.makedirs(d)

    return fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Which dataset', default='sample_sparse_dataset')
    parser.add_argument('--limit', help='Limit', type=int, default=-1)
    parser.add_argument('--algo', help='run only this algorithm', default=None)
    parser.add_argument('--pivot_file', help='pivot file for napp (nmslib)', default=None)

    args = parser.parse_args()

    results_fn = get_fn('results', args)
    queries_fn = get_fn('queries', args)

    print('storing queries in {} and results in {}'.format(queries_fn, results_fn))

    if not os.path.exists(queries_fn):
        queries = get_queries(args)
        with open(queries_fn, 'wb') as fd:
            pickle.dump(queries, fd)
    else:
        with open(queries_fn) as fd:
            queries = pickle.load(fd)

    print('got {} queries'.format(len(queries)))

    algos_already_ran = set()
    if os.path.exists(results_fn):
        with open(results_fn) as fd:
            for line in fd:
                library, algo_name = line.strip().split('\t')[:2]
                algos_already_ran.add((library, algo_name))

    algos = get_algos(args.pivot_file)

    if args.algo:
        print('running only {}'.format(args.algo))
        algos = {args.algo: algos[args.algo]}

    algos_flat = []

    for library in algos.keys():
        for algo in algos[library]:
            if (library, algo.name) not in algos_already_ran:
                algos_flat.append((library, algo))

    random.shuffle(algos_flat)

    print('order: {}'.format([a.name for l, a in algos_flat]))

    for library, algo in algos_flat:
        print('{}...'.format(algo.name))
        # Spawn a subprocess to force the memory to be reclaimed at the end
        p = multiprocessing.Process(target=run_algo,
                                    args=(args, library, algo, queries, results_fn))
        p.start()
        p.join()
