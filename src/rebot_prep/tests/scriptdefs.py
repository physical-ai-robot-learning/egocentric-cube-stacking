"""Load selected top-level definitions (functions, classes, constant assignments) of a script module whose import runs the
whole program, without running it: the pre-rewrite characterization of functions that were not importable."""
from __future__ import annotations

import ast
from pathlib import Path


def loadDefs(path: Path, names: set[str], namespace: dict) -> dict:
    tTree = ast.parse(Path(path).read_text())
    tKeep = []
    for tNode in tTree.body:
        if isinstance(tNode, (ast.FunctionDef, ast.ClassDef)) and tNode.name in names:
            tKeep.append(tNode)
        elif isinstance(tNode, ast.Assign) and any(isinstance(n, ast.Name) and n.id in names for t in tNode.targets for n in ast.walk(t)):
            tKeep.append(tNode)
        elif isinstance(tNode, (ast.Import, ast.ImportFrom)) and not (isinstance(tNode, ast.ImportFrom) and tNode.module == "ego_paths"):
            tKeep.append(tNode)
    tModule = ast.Module(body=tKeep, type_ignores=[])
    exec(compile(tModule, str(path), "exec"), namespace)
    return namespace
