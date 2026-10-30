"""Split sessions into train and test before anything is learned from them.

The split is stratified by chief complaint: within each chief complaint, 20% of the sessions
go to the test set, so a common complaint cannot end up almost only in one of the two sets.
Chief complaints with fewer than 5 sessions are pooled into one group, so rare complaints
are tested too. A session that is an exact copy of an earlier row goes wherever that row goes,
so the same session is never in both sets.
"""
import json
import random

TEST_SHARE = 0.2
SEED = 42
MIN_GROUP_SIZE = 5
RARE = "(rare)"


def split_sessions(sessions, test_share=TEST_SHARE, seed=SEED):
    """Returns (train rows, test rows): two sorted lists of CSV row numbers."""
    originals = []
    for s in sessions:
        if s.copy_of is None:
            originals.append(s)

    counts = {}
    for s in originals:
        cc = s.chief_complaint or RARE
        counts[cc] = counts.get(cc, 0) + 1

    # chief complaint -> its row numbers
    groups = {}
    for s in originals:
        cc = s.chief_complaint or RARE
        if counts[cc] < MIN_GROUP_SIZE:
            cc = RARE
        groups.setdefault(cc, []).append(s.row)

    # The same seed always gives the same split.
    rng = random.Random(seed)
    train = []
    test = []
    for cc in sorted(groups):
        rows = groups[cc]
        rng.shuffle(rows)
        n_test = round(len(rows) * test_share)
        test += rows[:n_test]
        train += rows[n_test:]

    # Copies follow their original.
    test_rows = set(test)
    for s in sessions:
        if s.copy_of is not None:
            if s.copy_of in test_rows:
                test.append(s.row)
            else:
                train.append(s.row)
    return sorted(train), sorted(test)


def save_split(path, train, test):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {"seed": SEED, "test_share": TEST_SHARE, "train": train, "test": test}
    path.write_text(json.dumps(content), encoding="utf-8")


def load_split(path):
    """Returns (train rows, test rows) saved by save_split."""
    content = json.loads(path.read_text(encoding="utf-8"))
    return content["train"], content["test"]
