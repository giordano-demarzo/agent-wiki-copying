"""The numbers quoted in "The platform and the population"."""
import collections

import numpy as np

from common import Dataset, Report, T


def main():
    say = Report('The platform and the population (Figure 1)')
    ds = Dataset()
    say(f'{len(ds.revs_all)} non-human edits under {len(ds.birth)} handles')
    say(f'population: {len(ds.handles)} handles, {len(ds.revs)} edits')
    task_revs = [r for r in ds.revs if ds.is_task(r)]
    task_pages = {r['page_id'] for r in task_revs}
    say(f'  of which {len(task_revs)} edits on {len(task_pages)} task pages '
        f'in {len(ds.task_families)} task families')
    say(f'  excluded: {len(ds.swarm)} handles, '
        f'{sum(1 for h in ds.swarm if ds.birth[h].strftime("%d %b") == "18 Jun")} born on 18 June')

    spans = [(T(v[-1]['time']) - T(v[0]['time'])).total_seconds() / 3600
             for v in ds.by_label.values() if len(v) >= 2]
    say(f'handle activity span (n={len(spans)} with at least two edits): '
        f'median {np.median(spans):.2f} h')
    say(f'edits per handle: mean {np.mean([len(v) for v in ds.by_label.values()]):.2f}')
    by_page = ds.by_page()
    page_spans = [(T(v[-1]['time']) - T(v[0]['time'])).total_seconds() / 3600
                  for v in by_page.values() if len(v) >= 2]
    say(f'page write span (n={len(page_spans)}): median {np.median(page_spans):.2f} h')
    say(f'pages touched by the population: {len(by_page)}')
    say.write()


if __name__ == '__main__':
    main()
