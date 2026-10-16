"""Fig. 4a, laboratory curve, for another model: P(agent opens the accurate source) and P(answers right) on a page of
N real notes from the record that all give the wrong value (N = 0, 1, 7, 19, 32), four OECD questions, 25 repetitions
(the design of real_support.py, style 'record', and check_probe.py allwrong N = 0).

    CHAIN_MODEL=gpt-6.1-sol CHAIN_BUDGET=15 python panel_a_model.py [N ...]   -> appends to out/panel_a_<model>.jsonl
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import chain_replay as cr  # noqa: E402
import facts_pilot as fp  # noqa: E402
from own_evidence import QUESTION  # noqa: E402
from real_support import COUNTRIES, page  # noqa: E402

NS = tuple(int(x) for x in sys.argv[1:] if x.isdigit()) or (0, 1, 7, 19, 32)   # page sizes to run (appended to the output)
REPS = 25
OUT = ROOT / 'data' / 'answers' / f'panel_a_{cr.MODEL}.jsonl'


async def one(client, country, n, rep, sem):
    notes = page(country, n, 'record', rep) if n else []
    run = cr.agent_responses if cr.MODEL in cr.PRICE else cr.agent
    a = await run(client, QUESTION.format(c=country), notes, 'probe_tool_ans', sem)
    v = fp.value(a['answer'])
    right, wrong = cr.COUNTRIES[country][:2]
    return dict(model=cr.MODEL, country=country, n=n, rep=rep, answer=a['answer'], is_right=v == float(right),
                is_wrong=v == float(wrong), checked=any(c['source'] for c in a['calls']), note=a['note'], error=a.get('error'))


async def run():
    client = cr.make_client()
    sem = asyncio.Semaphore(16 if cr.MODEL in cr.PRICE else 64)
    jobs = [(c, n, r) for n in NS for c in COUNTRIES for r in range(REPS)]
    res = await asyncio.gather(*[one(client, *j, sem) for j in jobs])
    with open(OUT, 'a') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')
    print(f'{len(res)} trials -> {OUT}; spent ${cr.SPENT[0]:.2f}')


if __name__ == '__main__':
    asyncio.run(run())
