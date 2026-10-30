"""Helpers for the DDx trees in a chapter file.

A tree is nested branches that end in lists of disease ids, for example
    {"viral": ["flu"], "bacterial": {"lung": ["pneumonia"]}}
A branch is named by its path from the tree name: "main.bacterial.lung".
"""


def all_branches(trees):
    """Every branch path -> all disease ids anywhere under it.

    {"main": {"viral": ["flu"], "bacterial": ["pneumonia"]}} gives
    {"main": ["flu", "pneumonia"], "main.viral": ["flu"], "main.bacterial": ["pneumonia"]}
    """
    result = {}
    for tree_name, node in trees.items():
        _collect(node, tree_name, result)
    return result


def _collect(node, path, result):
    if isinstance(node, list):
        result[path] = list(node)
        return result[path]
    diseases = []
    for key, child in node.items():
        diseases += _collect(child, path + "." + key, result)
    result[path] = diseases
    return diseases


def leaf_lists(trees):
    """Only the branch paths that hold a list of diseases directly."""
    result = {}
    for tree_name, node in trees.items():
        _collect_leaves(node, tree_name, result)
    return result


def _collect_leaves(node, path, result):
    if isinstance(node, list):
        result[path] = node
    else:
        for key, child in node.items():
            _collect_leaves(child, path + "." + key, result)


def shape_problems(trees):
    """A tree may only contain branches (mappings) and lists of disease ids."""
    problems = []
    for tree_name, node in trees.items():
        _check_shape(node, tree_name, problems)
    return problems


def _check_shape(node, path, problems):
    if isinstance(node, list):
        for item in node:
            if not isinstance(item, str):
                problems.append(f"{path}: a leaf must be a list of disease ids")
    elif isinstance(node, dict):
        for key, child in node.items():
            _check_shape(child, path + "." + key, problems)
    else:
        problems.append(f"{path}: expected branches or a list of disease ids")
