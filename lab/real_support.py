"""Strong support for the wrong value, accurate source. Single agents (gpt-oss-20b) see an all-wrong page of N
notes and have a fetch tool whose source gives the stored value (so checking always yields the right answer).
Page styles:
    definitive  template notes claiming a definitive source ('Dashboard workbook definitive value: ...')
    record      real notes from the record, written before the first correction, that give the padded value
                of the agent's own question (sampled without replacement, random order)
(the plain template notes are check_probe.py allwrong). N = 1, 7, 19, 32; Czech Republic, Hungary, Poland,
Slovak Republic (Slovenia has only 15 real notes); 25 repetitions.

    python real_support.py run   -> out/real_support_gpt-oss-20b.jsonl
"""
import asyncio
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import chain_replay as cr  # noqa: E402
import facts_pilot as fp  # noqa: E402
from own_evidence import DEFINITIVE, QUESTION  # noqa: E402
from check_probe import MONTHS, PREFIX, ROLE  # noqa: E402

OUT = ROOT / 'data' / 'answers' / 'real_support_gpt-oss-20b.jsonl'
NS, REPS = (1, 7, 19, 32), 25
COUNTRIES = ('Czech Republic', 'Hungary', 'Poland', 'Slovak Republic')
REAL = json.load(open(ROOT / 'sources' / 'record_padded_notes.json'))


def page(country, n, style, rep):
    wrong = cr.COUNTRIES[country][1]
    rng = random.Random(f'support|{country}|{n}|{style}|{rep}')
    if style == 'record':
        pool = [x for x in REAL if re.search(r'(?<![\d.])' + re.escape(wrong) + r'(?!\d)', x)]
        return rng.sample(pool, n)
    return [rng.choice(DEFINITIVE).format(c=country, v=wrong) +
            f' -- {rng.choice(PREFIX)}{rng.choice(MONTHS)}{rng.randint(1, 28):02d}{rng.choice(ROLE)}' for _ in range(n)]


async def one(client, country, n, style, rep, sem):
    a = await cr.agent(client, QUESTION.format(c=country), page(country, n, style, rep), 'probe_tool_ans', sem)
    v = fp.value(a['answer'])
    right, wrong = cr.COUNTRIES[country][:2]
    return dict(country=country, n=n, style=style, rep=rep, answer=a['answer'], is_right=v == float(right),
                is_wrong=v == float(wrong), checked=any(c['source'] for c in a['calls']), note=a['note'], error=a.get('error'))


async def run():
    client = cr.make_client()
    sem = asyncio.Semaphore(64)
    jobs = [(c, n, s, r) for c in COUNTRIES for n in NS for s in ('definitive', 'record') for r in range(REPS)]
    res = await asyncio.gather(*[one(client, *j, sem) for j in jobs])
    with open(OUT, 'w') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')
    print(f'{len(res)} trials -> {OUT}')


if __name__ == '__main__':
    asyncio.run(run())
