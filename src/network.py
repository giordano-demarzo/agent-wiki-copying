"""The network of handles for Figure 1c. Writes cache/network.json.

Two handles are linked if they edited at least one task page in common. Only the
largest connected component is laid out; the layout is Fruchterman-Reingold with a
fixed seed. Each handle carries the task of its first task edit.
"""
import collections
import itertools
import json
import os

import numpy as np

from common import CACHE, Dataset, Report

N_ITER = 500
SEED = 1


def components(nodes, edges):
    parent = {n: n for n in nodes}

    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for u, v in edges:
        parent[root(u)] = root(v)
    comp = collections.defaultdict(list)
    for n in nodes:
        comp[root(n)].append(n)
    return sorted(comp.values(), key=len, reverse=True)


def layout(n, edges, iterations=N_ITER, seed=SEED):
    """Fruchterman-Reingold on a dense adjacency, as in networkx.spring_layout."""
    rng = np.random.RandomState(seed)
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] = A[v, u] = 1
    pos = rng.rand(n, 2)
    k = np.sqrt(1 / n)
    t = max(np.ptp(pos, 0)) * 0.1
    dt = t / (iterations + 1)
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=-1), 0.01)
        force = k * k / dist ** 2 - A * dist / k
        disp = np.einsum('ijk,ij->ik', delta, force)
        length = np.maximum(np.linalg.norm(disp, axis=-1), 0.01)
        pos += disp * (t / length)[:, None]
        t -= dt
    return pos


def main():
    say = Report('The network of handles (Figure 1c)')
    ds = Dataset()
    editors = [sorted({r['label'] for r in rs}) for rs in ds.by_page(task_only=True).values()]
    links = sorted({pair for hs in editors for pair in itertools.combinations(hs, 2)})
    linked = sorted({h for pair in links for h in pair})
    parts = components(linked, links)
    giant = parts[0]
    index = {h: i for i, h in enumerate(giant)}
    edges = [(index[u], index[v]) for u, v in links if u in index]
    pos = layout(len(giant), edges)

    out = {'nodes': [{'id': h, 'x': float(x), 'y': float(y), 'task': ds.first_family[h]}
                     for h, (x, y) in zip(giant, pos)],
           'edges': edges}
    with open(os.path.join(CACHE, 'network.json'), 'w') as fh:
        json.dump(out, fh)

    say(f'{len(ds.handles)} handles, {len(links)} links (a task page edited in common)')
    say(f'  {len(ds.handles) - len(linked)} handles share no task page with anyone')
    say(f'  largest component: {len(giant)} handles, {len(edges)} links; '
        f'the other {len(linked) - len(giant)} linked handles are in {len(parts) - 1} small components')
    say.write()


if __name__ == '__main__':
    main()
