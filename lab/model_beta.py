"""The copy model with S-shaped copying (respond(rho, beta) in place of rho): the prior-strength sweep at the
record's |beta| (names, 1.68) and a GPT-6.1-Sol-like |beta| (3.9), 50 runs each.

    python model_beta.py   -> cache/model_beta.pkl
"""
import multiprocessing
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'src'))
import model as md  # noqa: E402
from common import load_cache  # noqa: E402

if __name__ == '__main__':
    par = load_cache('model.pkl')['par']
    with multiprocessing.Pool(60) as pool:
        out = {b: md.sweep(par, pool, beta=b) for b in (1.68, 3.9)}
    pickle.dump(out, open(ROOT.parent / 'cache' / 'model_beta.pkl', 'wb'))
    print('done')
