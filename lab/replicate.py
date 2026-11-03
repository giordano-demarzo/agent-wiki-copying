"""Fig. 4a and 4b for other models: the designs of real_support.py / panel_a_model.py (design A) and
check_probe.py (design B), with the same pages, seeds, prompt, fetch tool and scoring as gpt-oss-20b, on any model
behind an OpenAI-compatible endpoint (here vLLM on localhost).

    A  pages of N notes of the record, all giving the padded (wrong) value, N = 0, 1, 3, 7, 11, 19, 32;
       Czech Republic, Hungary, Poland, Slovak Republic; 25 agents per country and N
    P  as A, with template notes (the wordings, handles and order of check_probe.page) instead of the record's:
       agreement without the authority the record's notes claim; N = 1, 3, 7, 11, 19, 32 (N = 0 is A's empty page)
    B  pages of 20 notes, a fraction 0, 0.1, ..., 1 of them giving the right value; template wordings, random
       handles and order (check_probe.page); five countries; 25 agents per country and fraction
    R  as B, with the record's notes: 20 notes drawn from the record's notes that give the padded value (and
       neither the right value nor the stored digits), a fraction of them turned right by writing the right value
       in place of the padded one, so that right and wrong notes differ only in the value; four countries
    M  as R at a fixed mix and a varying length: N of the record's notes, half of them turned right,
       N = 2, 4, 8, 12, 20, 32 (does checking fall with the length of the page when the page disagrees with itself?)

Every agent: the OECD Equity question, the page, the fetch tool (answers_run.fetch: the dashboard page names the
workbook, the workbook returns the stored values), up to eight model turns, a JSON reply with the answer and a
note. An agent checks if it fetched the workbook. Each record keeps the design cell, the answer and its scoring,
the URLs fetched, the tokens used, and the raw final reply when it could not be parsed.

    python replicate.py --model gemma-4-31b                      # designs A and B, resumable
    python replicate.py --model gemma-4-31b --pilot 5            # 5 agents per country at N = 0, 7, 32 and frac 0, 0.5, 1
    python replicate.py --model qwen3.8-27b --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
    CHAIN_MODEL=gpt-6.1-sol CHAIN_BUDGET=25 python replicate.py --model gpt-6.1-sol --designs BPR
                         # an API model: through chain_replay.agent_responses (the Responses API, reasoning effort
                         # low, as in the earlier Sol runs), stopped at the budget in dollars
Writes data/answers/replicate_<model>.jsonl (or replicate_pilot_<model>.jsonl).
"""
import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
_argv, sys.argv = sys.argv, [sys.argv[0], 'replicate']   # the imported probes read sys.argv at import
import answers_run as ar  # noqa: E402
import chain_replay as cr  # noqa: E402
import check_probe as cp  # noqa: E402
import facts_pilot as fp  # noqa: E402
from own_evidence import QUESTION  # noqa: E402
from real_support import COUNTRIES as C4, page as record_page  # noqa: E402
sys.argv = _argv

NS = (0, 1, 3, 7, 11, 19, 32)
NS_M = (2, 4, 8, 12, 20, 32)
STORED = {c: v[3][:5] for c, v in cr.COUNTRIES.items()}   # the first digits of the stored value, e.g. '9.694'


def padded(v):
    return r'(?<![\d.])' + re.escape(v) + r'(?!\d)'


def real_mix(country, frac, rep, n=20):
    """R: n of the record's padded notes, a fraction frac of them turned right."""
    right, wrong = cr.COUNTRIES[country][:2]
    pool = [x for x in json.load(open(ROOT / 'sources' / 'record_padded_notes.json'))
            if re.search(padded(wrong), x) and not re.search(padded(right), x) and STORED[country] not in x]
    rng = random.Random(f'realmix|{country}|{frac}|{rep}' + (f'|{n}' if n != 20 else ''))   # R keeps its seeds
    notes = rng.sample(pool, n)
    flip = set(rng.sample(range(n), round(frac * n)))
    return [re.sub(padded(wrong), right, x) if i in flip else x for i, x in enumerate(notes)]


def template_page(country, n, rep):
    """P: n template notes, all giving the padded value (check_probe.page at frac 0, any length)."""
    keep = cp.N_NOTES
    cp.N_NOTES = n
    try:
        return cp.page(country, 0.0, rep)
    finally:
        cp.N_NOTES = keep
FRACS = tuple(round(0.1 * i, 1) for i in range(11))
REPS = 25
assert cp.N_NOTES == 20


def cells(designs, pilot, fracs_b=None):
    """(design, country, n or frac, rep) for every agent to run."""
    out = []
    if 'A' in designs:
        for n in ((0, 7, 32) if pilot else NS):
            out += [('A', c, n, r) for c in C4 for r in range(pilot or REPS)]
    if 'B' in designs:
        for f in ((0.0, 0.5, 1.0) if pilot else (fracs_b or FRACS)):
            out += [('B', c, f, r) for c in cr.COUNTRIES for r in range(pilot or REPS)]
    if 'P' in designs:
        for n in ((7, 32) if pilot else NS[1:]):
            out += [('P', c, n, r) for c in C4 for r in range(pilot or REPS)]
    if 'R' in designs:
        for f in ((0.0, 0.5, 1.0) if pilot else FRACS):
            out += [('R', c, f, r) for c in C4 for r in range(pilot or REPS)]
    if 'M' in designs:
        for n in ((8, 32) if pilot else NS_M):
            out += [('M', c, n, r) for c in C4 for r in range(pilot or REPS)]
    return out


def page(design, country, x, rep):
    if design == 'A':
        return record_page(country, x, 'record', rep) if x else []
    if design == 'P':
        return template_page(country, x, rep)
    if design == 'R':
        return real_mix(country, x, rep)
    if design == 'M':
        return real_mix(country, 0.5, rep, n=x)
    return cp.page(country, x, rep)


def parse(raw):
    """The reply as JSON; failing that (bold markers, unquoted numbers, trailing code), the answer field read with a
    pattern. 'json' says whether the reply was valid JSON."""
    mm = re.search(r'\{.*\}', raw, re.S)
    try:
        p = json.loads(mm.group(0)) if mm else {}
        if isinstance(p, dict) and p:
            return dict(p, json=True)
    except json.JSONDecodeError:
        pass
    m = re.search(r'"answer"\s*:\s*"?\**\s*([0-9]+(?:\.[0-9]+)?)', raw)
    n = re.search(r'"wiki_note"\s*:\s*"?(.*)', raw, re.S)
    return dict(answer=m.group(1) if m else None, wiki_note=n.group(1)[:2000] if n else '', json=False)


async def api_agent(client, question, notes, sem):
    """An API model (GPT-6.1 Sol): the loop of its earlier runs, mapped onto the fields of agent()."""
    res = await cr.agent_responses(client, question, notes, 'probe_tool_ans', sem)
    return dict(answer=res.get('answer'), note=res.get('note', ''), urls=[c['url'] for c in res.get('calls', [])],
                turns=None, tokens_in=None, tokens_out=None, raw=None, error=res.get('error'))


async def agent(client, a, question, notes, sem):
    if cr.MODEL in cr.PRICE:
        assert a.model == cr.MODEL, 'set CHAIN_MODEL to the API model'
        return await api_agent(client, question, notes, sem)
    msgs = [{'role': 'system', 'content': cr.SYSTEM}, {'role': 'user', 'content': cr.prompt(question, notes, True, True, cr.NOTE_ANS)}]
    kw = dict(model=a.model, temperature=a.temperature, max_tokens=a.max_tokens, tools=ar.TOOLS)
    if a.reasoning_effort:
        kw['reasoning_effort'] = a.reasoning_effort
    if a.extra_body:
        kw['extra_body'] = json.loads(a.extra_body)
    urls, tok_in, tok_out, raw = [], 0, 0, ''
    async with sem:
        try:
            for turn in range(8):
                r = await client.chat.completions.create(messages=msgs, **kw)
                tok_in += r.usage.prompt_tokens
                tok_out += r.usage.completion_tokens
                m = r.choices[0].message
                if m.tool_calls:
                    msgs.append({'role': 'assistant', 'content': m.content or '',
                                 'tool_calls': [tc.model_dump() for tc in m.tool_calls]})
                    for tc in m.tool_calls:
                        try:
                            url = json.loads(tc.function.arguments).get('url', '')
                        except (json.JSONDecodeError, AttributeError):
                            url = ''
                        urls.append(url)
                        msgs.append({'role': 'tool', 'tool_call_id': tc.id, 'content': ar.fetch(url)[:60000]})
                    continue
                raw = m.content or ''
                p = parse(raw)
                return dict(answer=p.get('answer'), note=str(p.get('wiki_note') or ''), urls=urls, turns=turn + 1,
                            tokens_in=tok_in, tokens_out=tok_out, raw=None if p.get('json') else raw[:3000],
                            error=None)
            error = 'max turns'
        except Exception as e:  # noqa: BLE001
            error = repr(e)[:300]
    return dict(answer=None, note='', urls=urls, turns=None, tokens_in=tok_in, tokens_out=tok_out, raw=raw[:3000] or None,
                error=error)


async def one(client, a, cell, sem, f, lock):
    design, country, x, rep = cell
    res = await agent(client, a, QUESTION.format(c=country), page(design, country, x, rep), sem)
    v = fp.value(res['answer'])
    right, wrong = cr.COUNTRIES[country][:2]
    rec = dict(model=a.model, design=design, country=country, rep=rep, **({'n': x} if design in 'APM' else {'frac_right': x}),
               is_right=v == float(right), is_wrong=v == float(wrong), checked=any('Data-Education-equity-dashboard' in u
                                                                                   for u in res['urls']), **res)
    async with lock:
        f.write(json.dumps(rec) + '\n')
        f.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='the served model name')
    ap.add_argument('--base-url', default='http://localhost:8011/v1')
    ap.add_argument('--designs', default='AB')
    ap.add_argument('--pilot', type=int, default=0, help='agents per country in a pilot (fewer cells)')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max-tokens', type=int, default=16000)
    ap.add_argument('--reasoning-effort', default=None)
    ap.add_argument('--extra-body', default=None, help='JSON passed to the server, e.g. chat_template_kwargs')
    ap.add_argument('--concurrency', type=int, default=64)
    ap.add_argument('--tag', default='', help='a suffix for the output file, e.g. a repeat or a variant of a run')
    ap.add_argument('--fracs-b', default=None, help='design B at these fractions only, e.g. 0,0.2,0.4,0.6,0.8,1')
    ap.add_argument('--retry', action='store_true',
                    help='run again the agents with an error or without a readable answer (set aside to _failed)')
    a = ap.parse_args()
    from openai import AsyncOpenAI
    client = cr.make_client() if cr.MODEL in cr.PRICE else AsyncOpenAI(base_url=a.base_url, api_key='EMPTY', timeout=900)
    out = ROOT / 'data' / 'answers' / f'replicate_{"pilot_" if a.pilot else ""}{a.model}{"_" + a.tag if a.tag else ""}.jsonl'
    done = set()
    if a.retry and out.exists():
        rows = [json.loads(l) for l in open(out)]
        bad = [r for r in rows if r.get('error') or r.get('answer') is None]
        with open(out.with_name(out.stem + '_failed.jsonl'), 'a') as f:
            for r in bad:
                f.write(json.dumps(r) + '\n')
        with open(out, 'w') as f:
            for r in rows:
                if not (r.get('error') or r.get('answer') is None):
                    f.write(json.dumps(r) + '\n')
        print(f'{len(bad)} agents set aside to run again')
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            if not r.get('error'):
                done.add((r['design'], r['country'], r.get('n', r.get('frac_right')), r['rep']))
    fracs_b = tuple(round(float(x), 1) for x in a.fracs_b.split(',')) if a.fracs_b else None
    todo = [c for c in cells(a.designs, a.pilot, fracs_b) if c not in done]
    print(f'{a.model}: {len(todo)} agents to run ({len(done)} done) -> {out}', flush=True)
    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    with open(out, 'a') as f:
        await asyncio.gather(*[one(client, a, c, sem, f, lock) for c in todo])
    if cr.MODEL in cr.PRICE:
        print(f'spent ${cr.SPENT[0]:.2f} of a ${cr.BUDGET:.0f} budget')
    summary(out)


def summary(path):
    rows = [json.loads(l) for l in open(path)]
    ok = [r for r in rows if not r['error']]
    print(f'{len(rows)} agents, {len(rows) - len(ok)} errors, {sum(r["raw"] is not None for r in ok)} unparsed replies')
    for key, d in (('n', 'A'), ('n', 'P'), ('n', 'M'), ('frac_right', 'B'), ('frac_right', 'R')):
        g = {}
        for r in ok:
            if r['design'] == d:
                g.setdefault(r[key], []).append(r)
        for x in sorted(g):
            rs = g[x]
            print(f'  {d} {key}={x}: n={len(rs)} checked {sum(r["checked"] for r in rs) / len(rs):.2f} '
                  f'right {sum(r["is_right"] for r in rs) / len(rs):.2f} wrong {sum(r["is_wrong"] for r in rs) / len(rs):.2f} '
                  f'tool use {sum(bool(r["urls"]) for r in rs) / len(rs):.2f}')


if __name__ == '__main__':
    asyncio.run(main())
