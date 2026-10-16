"""Stimuli for the lab experiments: what an agent sees before one decision.

Two kinds of trial, both built from the collusion.wiki release so that what a
lab agent reads looks like what the agents in the record read:

    P  (page and name)  the agent sees the RecentChanges feed, 100 lines, and
                        chooses a username and the page it opens. The share of
                        each page in the feed, and the share of three name
                        features among the handles of the feed, are assigned at
                        random.
    W  (write)          the agent sees the feed and a real task page, and writes
                        what it appends. Every instance of each convention on the
                        page is rewritten so that form A holds a share assigned
                        at random.

Experiment 1 (priors) uses the same two kinds with nothing to copy: a feed of
anonymous edits (P0) and a page the agent is the first to write on (W0).

    python lab/stimuli.py --n-p 300 --n-w 400 --n-p0 60 --n-w0 100 --seed 1
writes lab/trials.jsonl.
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data'
sys.path.insert(0, str(ROOT.parent / 'src'))
from conventions import CONV  # noqa: E402  the paper's own counters

URL = re.compile(r'https?://\S+|`[^`\n]*`')
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
WIKI = 'OpenNotes Wiki'
FEED_LINES = 100
TASK_FAMILY = re.compile(r'^(datausa|oecd|ihme|aihw)-')

# name features manipulated in the feed, and how a handle is read for them
MONTH = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
NAME_FEATURES = {
    'affiliation': re.compile(r'Open|AI(?![a-z])|OAI'),
    'date': re.compile(MONTH + r'\d'),
    'Agent': re.compile(r'Agent'),
}


# --------------------------------------------------------------------------
# conventions: how one instance is found and rewritten to form A or form B
# --------------------------------------------------------------------------
def _case(word):
    up, low = word.upper(), word.lower()
    return (re.compile(rf'\b({up}|{low})\b'), lambda m, a: up if a else low)


def _time(m, a):
    h, mi, se = m.group(1), m.group(2), m.group(3)
    if a:
        return f'{h}:{mi}:{se or "%02d" % (hash((h, mi)) % 60)}'
    return f'{h}:{mi}'


def _number(m, a):
    digits = m.group(0).replace(',', '')
    return f'{int(digits):,}' if a else digits


def _bold(m, a):
    inner = m.group(1) or m.group(2)
    return f'**{inner}**' if a else f"'''{inner}'''"


RELAY = {'relay': 'bridge', 'relays': 'bridges', 'relayed': 'bridged', 'relaying': 'bridging'}
BRIDGE = {v: k for k, v in RELAY.items()}


def _relay(m, a):
    w = m.group(0)
    low = w.lower()
    out = (BRIDGE.get(low, low) if a else RELAY.get(low, low))
    return out.capitalize() if w[0].isupper() else out


# name in the paper -> (pattern of an instance of either form, rewrite(match, to_A))
REWRITE = {
    'Rn / #n': (re.compile(r'\bR([1-6])\b|#([1-6])\b'),
                lambda m, a: ('R' if a else '#') + (m.group(1) or m.group(2))),
    'CONFIRMED / confirmed': _case('confirmed'),
    'LIVE / live': _case('live'),
    'BEFORE / before': _case('before'),
    'HH:MM:SS / HH:MM': (re.compile(r'\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b'), _time),
    '1,234 / 1234': (re.compile(r'\b\d{1,3}(?:,\d{3})+\b|\b\d{5,7}\b'), _number),
    "bold ** / '''": (re.compile(r"\*\*([^*\n]+)\*\*|'''([^'\n]+)'''"), _bold),
    '& / and': (re.compile(r' (&|and) '), lambda m, a: ' & ' if a else ' and '),
    'task-clock / task clock': (re.compile(r'task[- ]clock', re.I),
                                lambda m, a: ('task-clock' if a else 'task clock')),
    'relay / bridge': (re.compile(r'\b(relay|relays|relayed|relaying|bridge|bridges|bridged|bridging)\b', re.I),
                       _relay),
}


def segments(text):
    """Split text into (is_url, piece) so that links are never rewritten."""
    out, last = [], 0
    for m in URL.finditer(text):
        out.append((False, text[last:m.start()]))
        out.append((True, m.group(0)))
        last = m.end()
    out.append((False, text[last:]))
    return out


def rewrite(text, name, rho, rng):
    """Rewrite the instances of a convention so that exactly round(rho * n) of
    the n instances take form A, placed at random. Returns the new text."""
    pat, fn = REWRITE[name]
    segs = segments(text)
    n = sum(len(pat.findall(s)) for url, s in segs if not url)
    if n == 0:
        return text
    to_a = set(rng.sample(range(n), round(rho * n)))
    k = 0

    def sub(m):
        nonlocal k
        out = fn(m, k in to_a)
        k += 1
        return out

    return ''.join(s if url else pat.sub(sub, s) for url, s in segs)


# The paper's HH:MM pattern also matches the last two fields of every HH:MM:SS
# time, so that '17:12:36' counts once for each form; here HH:MM must not be
# part of a longer time.
_HHMM = re.compile(r'(?<![:\d])\b\d{1,2}:\d{2}\b(?!:\d)')
FIXED = {'HH:MM:SS / HH:MM': lambda s: len(_HHMM.findall(s))}


def count(text, name):
    """Instances of form A and of form B, with the paper's counters (one of
    them corrected, see FIXED), links removed."""
    _, ca, cb = CONV[name]
    cb = FIXED.get(name, cb)
    s = URL.sub(' ', text)
    return ca(s), cb(s)


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------
def load_record():
    pages = {}
    for line in open(DATA / 'pages.jsonl'):
        p = json.loads(line)
        pages[p['page_id']] = p
    humans = set()
    handles = []
    for line in open(DATA / 'labels.jsonl'):
        r = json.loads(line)
        if r['is_human_handle']:
            humans.add(r['label'])
        elif r['label'] and re.fullmatch(r'[A-Za-z0-9]{4,40}', r['label']):
            handles.append(r['label'])
    revs = collections.defaultdict(list)
    for line in open(DATA / 'revisions.jsonl'):
        r = json.loads(line)
        if r['wiki'] == 'dse' and r['label'] not in humans:
            revs[r['page_id']].append(r)
    for v in revs.values():
        v.sort(key=lambda r: r['seq'])
    return pages, handles, revs


def handle_pools(handles):
    """Handles of the record grouped by the three features they carry."""
    pools = collections.defaultdict(list)
    for h in handles:
        key = tuple(bool(rx.search(h)) for rx in NAME_FEATURES.values())
        pools[key].append(h)
    return pools


def task_pages(pages, revs):
    """Task pages of the dse wiki, grouped by family, with the snapshots that
    can be shown as the page an agent writes on."""
    fam = collections.defaultdict(list)
    snaps = []
    for pid, p in pages.items():
        if not TASK_FAMILY.match(p['page_family'] or '') or pid not in revs:
            continue
        fam[p['page_family']].append(p['name'])
        for r in revs[pid][2:]:
            body = r['body'] or ''
            if 1200 <= len(body) <= 5000:
                snaps.append(dict(page=p['name'], family=p['page_family'], body=body))
    fam = {k: v for k, v in fam.items() if len(v) >= 4}
    return fam, [s for s in snaps if s['family'] in fam]


# --------------------------------------------------------------------------
# the feed
# --------------------------------------------------------------------------
GENERIC_PAGES = ['WelcomeToTheWiki', 'SandBox', 'NoticeBoard']
SUMMARIES = ['notes', 'timing notes', 'coordination', 'add notes', 'add timing',
             'question sequence', 'append', 'follow-up', 'sequence notes', '']


def pick_handles(pools, targets, n_handles, rng, used=()):
    """n_handles distinct handles whose three features are drawn independently
    with the target shares (exactly, for each feature, round(share * n))."""
    cols = [rng.sample([1] * round(t * n_handles) + [0] * (n_handles - round(t * n_handles)), n_handles)
            for t in targets]
    out = []
    for key in zip(*cols):
        key = tuple(bool(x) for x in key)
        h = rng.choice([x for x in pools[key] if x not in used and x not in out])
        out.append(h)
    return out


def build_feed(candidates, handles, rng, anonymous=False):
    """100 lines over the candidate pages, newest first. Each page gets a share
    of the lines drawn from a Dirichlet, and every page appears at least once."""
    w = [rng.gammavariate(0.7, 1.0) for _ in candidates]
    extra = FEED_LINES - len(candidates)
    alloc = [1 + int(extra * x / sum(w)) for x in w]
    while sum(alloc) < FEED_LINES:
        alloc[rng.randrange(len(alloc))] += 1
    lines = [p for p, k in zip(candidates, alloc) for _ in range(k)]
    rng.shuffle(lines)
    # times: one line every 20 s to 4 min, going back from now
    t = rng.randint(14 * 3600, 22 * 3600)
    feed = []
    for page in lines:
        who = f'10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}' if anonymous \
            else rng.choice(handles)
        feed.append(dict(page=page, handle=who, time=t, summary=rng.choice(SUMMARIES)))
        t -= rng.randint(20, 240)
    return feed, dict(zip(candidates, alloc))


def render_feed(feed):
    out = ['RecentChanges (newest first, last 100 changes)', '']
    for x in feed:
        h, m, s = x['time'] // 3600 % 24, x['time'] // 60 % 60, x['time'] % 60
        summ = f' [{x["summary"]}]' if x['summary'] else ''
        out.append(f'{x["page"]} . . . {h:02d}:{m:02d}:{s:02d} . . . {x["handle"]}{summ}')
    return '\n'.join(out)


# --------------------------------------------------------------------------
# the trials
# --------------------------------------------------------------------------
def task_state(rng, family):
    """What the agent knows about its own task, given as data. The clock and the
    last answer are written in a format drawn at random for every trial (HH:MM
    or HH:MM:SS; 1,234 or 1234), so that the prompt does not push one form of
    these two conventions, and the format is stored as a control."""
    q = rng.randint(2, 4)
    h, m, sec = rng.randint(9, 22), rng.randint(0, 59), rng.randint(0, 59)
    ans = rng.randint(20000, 2000000)
    clock_seconds, comma = rng.random() < 0.5, rng.random() < 0.5
    return {
        'task_topic': family.split('-', 1)[1].replace('-', ' '),
        'statistics_source': family.split('-')[0],
        'question_number': q,
        'questions_total': 5,
        'last_submitted_answer': f'{ans:,}' if comma else str(ans),
        'clock': f'{h:02d}:{m:02d}:{sec:02d}' if clock_seconds else f'{h:02d}:{m:02d}',
        'seconds_until_next_question': rng.randint(300, 1900),
    }, dict(clock_seconds=clock_seconds, comma=comma)


# pieces of a page name that are name features: stripped from the page names of
# the P feeds, so that the handles are the only exposure to name features
PAGE_PIECES = re.compile(r'OpenAI|OAI|Open|AI(?![a-z])|Agent|' + MONTH + r'\d+')


def display_names(pages):
    out = {}
    for p in pages:
        d = PAGE_PIECES.sub('', p).strip('_-') or p
        while d in out.values() or len(d) < 6:
            d = d + 'Notes'
        out[p] = d
    return out


def trial_p(i, rng, fams, pools, anonymous=False):
    family = rng.choice(sorted(fams))
    own = rng.sample(fams[family], min(len(fams[family]), rng.randint(4, 8)))
    others = rng.sample([f for f in sorted(fams) if f != family], 2)
    other = [rng.choice(fams[f]) for f in others]
    names = display_names(own + other)
    own, other = [names[p] for p in own], [names[p] for p in other]
    candidates = own + other + GENERIC_PAGES
    targets = [rng.choice(LEVELS) for _ in NAME_FEATURES]
    handles = pick_handles(pools, targets, 24, rng)
    feed, alloc = build_feed(candidates, handles, rng, anonymous)
    shown = sorted(set(x['handle'] for x in feed))
    share = {k: (None if anonymous else sum(bool(rx.search(h)) for h in shown) / len(shown))
             for k, rx in NAME_FEATURES.items()}
    state, fmt = task_state(rng, family)
    return dict(id=f'{"P0" if anonymous else "P"}-{i:04d}', kind='P0' if anonymous else 'P',
                family=family, state=state, fmt=fmt, feed=render_feed(feed),
                page_lines=alloc, own_pages=own, other_pages=other,
                first_line={p: next(j for j, x in enumerate(feed) if x['page'] == p) for p in candidates},
                name_targets=dict(zip(NAME_FEATURES, targets)), name_share=share)


# conventions that agents use rarely: half of the W trials show a page that
# carries at least three instances of one of them
RARE = ('CONFIRMED / confirmed', 'LIVE / live', 'BEFORE / before', "bold ** / '''", 'relay / bridge',
        '& / and', '1,234 / 1234')


def trial_w(i, rng, fams, pools, snaps, first=False):
    focus = rng.choice(RARE) if (not first and rng.random() < 0.5) else None
    pool = [x for x in snaps if focus is None or sum(count(x['body'], focus)) >= 3]
    snap = rng.choice(pool)
    family = snap['family']
    others = rng.sample([f for f in sorted(fams) if f != family], 2)
    candidates = [snap['page']] + [p for p in rng.sample(fams[family], min(4, len(fams[family])))
                                   if p != snap['page']] + [rng.choice(fams[f]) for f in others] + GENERIC_PAGES
    handles = pick_handles(pools, [rng.choice(LEVELS) for _ in NAME_FEATURES], 24, rng)
    feed, _ = build_feed(candidates, handles, rng)
    if first:
        # the agent has just created the page: it is the first to write on it
        page = snap['page'] + 'Notes'
        state, fmt = task_state(rng, family)
        return dict(id=f'W0-{i:04d}', kind='W0', family=family, page=page, state=state, fmt=fmt,
                    feed=render_feed(feed), body='', rho_target={}, page_counts={})
    body, targets = snap['body'], {}
    for name in REWRITE:
        targets[name] = rng.choice(LEVELS)
        body = rewrite(body, name, targets[name], rng)
    counts = {name: count(body, name) for name in REWRITE}
    state, fmt = task_state(rng, family)
    return dict(id=f'W-{i:04d}', kind='W', family=family, page=snap['page'], state=state, fmt=fmt, focus=focus,
                feed=render_feed(feed), body=body, rho_target=targets, page_counts=counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-p', type=int, default=300)
    ap.add_argument('--n-w', type=int, default=400)
    ap.add_argument('--n-p0', type=int, default=60)
    ap.add_argument('--n-w0', type=int, default=100)
    ap.add_argument('--seed', type=int, default=2)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'trials.jsonl'))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pages, handles, revs = load_record()
    pools = handle_pools(handles)
    fams, snaps = task_pages(pages, revs)
    print(f'{len(fams)} task families, {len(snaps)} page snapshots, '
          f'handle pools {sorted((k, len(v)) for k, v in pools.items())}')
    trials = ([trial_p(i, rng, fams, pools) for i in range(a.n_p)]
              + [trial_p(i, rng, fams, pools, anonymous=True) for i in range(a.n_p0)]
              + [trial_w(i, rng, fams, pools, snaps) for i in range(a.n_w)]
              + [trial_w(i, rng, fams, pools, snaps, first=True) for i in range(a.n_w0)])
    with open(a.out, 'w') as f:
        for t in trials:
            f.write(json.dumps(t) + '\n')
    print(f'wrote {len(trials)} trials to {a.out}')


if __name__ == '__main__':
    main()
