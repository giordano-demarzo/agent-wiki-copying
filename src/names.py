"""Names: their pieces, the classification of pieces, the 16 name features, and
what a newcomer could see of the names of others when it arrived.

A name is a string of capitalised pieces (OpenAIResearchHelperMay08 gives
Open, AI, Research, Helper, May). Every piece carried by at least 15 handles is
classified by one number, its over-representation in a single task: the share
of its carriers in its most common task family, divided by that family's
share of the population. Months form the date stamp. Pieces over-represented
at least four-fold belong to one task; the others are generic. The generic
pieces and the date stamp are the name features used throughout the paper.

The exposure of a newcomer is defined once, here, and used by every analysis:
  page authors  the handles that had edited, before the newcomer's first edit,
                any page the newcomer ever writes on;
  feed          the handles of the last 100 feed lines before its first edit;
  task-mates    the handles of its own task family born in the previous six
                hours; a task-mate is in view if it is a page author or in the
                feed, and out of view otherwise.
"""
import collections
import re

import numpy as np

from common import FEED_LINES, T, block_of, coauthor_matrix, co_occurrence_coauthors, share, volatility
from conventions import fit_mu

PIECE = re.compile(r'[A-Z][a-z]+|[A-Z]{2,}(?![a-z])')
MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
DATE = re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d')
AFFILIATION = ('Open', 'AI', 'OAI')
MIN_CARRIERS = 15        # a piece is classified if at least this many handles carry it
TASK_SPECIFIC = 4.0      # over-representation in one task from which a piece belongs to it
MATES_H = 6              # task-mates are the handles of the same task born in the last 6 h
MIN_MATES = 6            # a newcomer enters the name analysis with at least this many task-mates
MIN_GROUP = 2            # a share is computed over at least this many handles
MIN_OTHER = 3            # ... and over at least this many for handles of other tasks
MIN_CARRIERS_COOCCURRENCE = 25   # co-occurrence among co-authors needs this many carriers


def pieces(handle):
    """The capitalised pieces of a name, without numbers."""
    return [p for p in PIECE.findall(handle) if not p.isdigit()]


def families_of(ds):
    """Handle index -> set of task families it wrote on."""
    out = collections.defaultdict(set)
    for r in ds.revs:
        if ds.is_task(r):
            out[ds.index[r['label']]].add(ds.family(r))
    return out


def classify_pieces(ds):
    """Every piece with at least MIN_CARRIERS carriers: carriers, its
    over-representation in its most common task family, and its class."""
    sets = [set(pieces(h)) for h in ds.handles]
    fam_of = families_of(ds)
    fam_size = collections.Counter(f for fs in fam_of.values() for f in fs)
    n_with = sum(1 for i in range(len(ds.handles)) if fam_of[i])
    counts = collections.Counter(p for s in sets for p in s)
    rows = []
    for p, c in counts.most_common():
        if c < MIN_CARRIERS:
            break
        carriers = [i for i in range(len(sets)) if p in sets[i] and fam_of[i]]
        per_family = collections.Counter(f for i in carriers for f in fam_of[i])
        top, k = per_family.most_common(1)[0]
        over = (k / len(carriers)) / (fam_size[top] / n_with)
        cls = 'month' if p in MONTHS else ('task-specific' if over >= TASK_SPECIFIC else 'generic')
        rows.append(dict(name=p, carriers=c, over=over, top_family=top, cls=cls))
    return rows


def features(ds, piece_rows):
    """The name features: every generic piece, and the date stamp. Returns
    name -> 0/1 indicator over the handles, in birth order."""
    sets = [set(pieces(h)) for h in ds.handles]
    out = {r['name']: np.array([r['name'] in s for s in sets], float)
           for r in piece_rows if r['cls'] == 'generic'}
    out['date'] = np.array([bool(DATE.search(h)) for h in ds.handles], float)
    return out


def indicators(ds, names):
    """0/1 indicator of a list of pieces (any of them) over the handles."""
    sets = [set(pieces(h)) for h in ds.handles]
    names = set(names)
    return np.array([bool(s & names) for s in sets], float)


def exposure_groups(ds, mates_h=MATES_H):
    """For every handle, in birth order, the index sets of what it could see:
    page authors, feed handles, and task-mates in and out of view."""
    labels = [r['label'] for r in ds.revs_all]
    pos_of_rev = {id(r): k for k, r in enumerate(ds.revs_all)}
    ts = np.array([T(r['time']).timestamp() for r in ds.revs_all])
    birth = np.array([ds.birth[h].timestamp() for h in ds.handles])
    fam = np.array([ds.first_family[h] for h in ds.handles])
    n = len(ds.handles)
    # every population edit of a page, as (position in the feed, handle index)
    edits_of_page = collections.defaultdict(list)
    for r in ds.revs:
        edits_of_page[r['page_id']].append((pos_of_rev[id(r)], ds.index[r['label']]))
    pages_of = [{r['page_id'] for r in ds.by_label[h]} for h in ds.handles]
    out = []
    for i in range(n):
        j = int(np.searchsorted(ts, birth[i]))
        authors = {h for pg in pages_of[i] for k, h in edits_of_page[pg] if k < j and h != i}
        feed = {ds.index[labels[k]] for k in range(max(0, j - FEED_LINES), j)
                if labels[k] in ds.index and ds.index[labels[k]] != i}
        mates = np.where((fam == fam[i]) & (birth < birth[i]) & (birth >= birth[i] - mates_h * 3600))[0]
        seen = authors | feed
        out.append(dict(
            enough_feed=j >= FEED_LINES,
            authors=authors, feed=feed,
            mates=mates,
            on_page=[m for m in mates if m in authors],
            in_feed=[m for m in mates if m in feed and m not in authors],
            in_view=[m for m in mates if m in seen],
            out_of_view=[m for m in mates if m not in seen],
            other_tasks=[h for h in feed if fam[h] != fam[i]],
            block=block_of(birth[i])))
    return out


def in_sample(g):
    """A newcomer enters the name analysis if the feed before it was complete
    and it had at least MIN_MATES task-mates."""
    return g['enough_feed'] and len(g['mates']) >= MIN_MATES


def records(ds, feats, groups):
    """One record per newcomer in the sample and name feature: the outcome and
    the share of the feature among the task-mates on the page, in the feed only,
    out of view, among all task-mates in view, and among other tasks in the feed."""
    fam = np.array([ds.first_family[h] for h in ds.handles])
    rows = []
    for i, g in enumerate(groups):
        if not in_sample(g):
            continue
        for t, v in feats.items():
            rows.append((t, i, fam[i], g['block'], v[i],
                         share(v, g['on_page'], MIN_GROUP), share(v, g['in_feed'], MIN_GROUP),
                         share(v, g['out_of_view'], MIN_GROUP), share(v, g['other_tasks'], MIN_OTHER),
                         share(v, g['in_view'], MIN_GROUP)))
    cols = list(zip(*rows))
    return dict(feature=np.array(cols[0]), handle=np.array(cols[1]), family=np.array(cols[2]),
                block=np.array(cols[3]), y=np.array(cols[4], float), page=np.array(cols[5], float),
                feed=np.array(cols[6], float), out=np.array(cols[7], float),
                other=np.array(cols[8], float), view=np.array(cols[9], float))


def coauthors(ds):
    """Co-authorship of task pages among the population's handles."""
    page_sets = collections.defaultdict(set)
    for r in ds.revs:
        if ds.is_task(r):
            page_sets[r['page_id']].add(ds.index[r['label']])
    return coauthor_matrix(page_sets.values(), len(ds.handles))


def feature_statistics(v, groups, A):
    """Prior strength (base probabilities fitted on the share among the
    task-mates in view), co-occurrence among co-authors, volatility over blocks
    of 100 newborns, and the overall share, for one 0/1 feature of handles."""
    x = np.array([share(v, g['in_view'], MIN_GROUP) if in_sample(g) else np.nan for g in groups])
    mu = fit_mu(x, v)
    return dict(muA=mu[0], muB=mu[1], s=min(mu[0] + mu[1], 1.0), f=v.mean(),
                gap=(co_occurrence_coauthors(v, A) if v.sum() >= MIN_CARRIERS_COOCCURRENCE else np.nan),
                vol=volatility(v, 100))
