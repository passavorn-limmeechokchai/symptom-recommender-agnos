"""Grouped folds inside training, balancing complaint and evaluation eligibility."""
import random
from collections import Counter, defaultdict


def training_folds(sessions, n_folds=5, seed=1729):
    if n_folds < 2:
        raise ValueError("At least two folds are required")
    groups = defaultdict(list)
    for session in sessions:
        groups[session.group()].append(session)
    if len(groups) < n_folds:
        raise ValueError("Not enough independent groups for the requested folds")
    complaints = Counter(group[0].chief_complaint for group in groups.values())
    strata = defaultdict(list)
    for group_id, group in groups.items():
        first = group[0]
        complaint = first.chief_complaint if complaints[first.chief_complaint] >= n_folds else "(rare)"
        strata[(complaint or "", bool(first.associated))].append(group_id)
    rng = random.Random(seed)
    folds = [set() for _ in range(n_folds)]
    for stratum in sorted(strata):
        group_ids = strata[stratum]
        rng.shuffle(group_ids)
        stratum_sizes = [0] * n_folds
        for group_id in group_ids:
            order = list(range(n_folds))
            rng.shuffle(order)
            chosen = min(order, key=lambda i: (stratum_sizes[i], len(folds[i])))
            folds[chosen].add(group_id)
            stratum_sizes[chosen] += 1
    return [([s for s in sessions if s.group() not in fold],
             [s for s in sessions if s.group() in fold]) for fold in folds]
