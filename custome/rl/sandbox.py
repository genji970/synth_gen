
import ast
import json
import os
import re


def bootstrap(*sources):
    allowed = {
        p.split("==")[0].replace("-", "_"): p
        for p in os.environ["RL_ALLOWED_PIP"].split()
    }
    aliases = {
        "sklearn": "scikit_learn",
        "PIL": "pillow",
        "cv2": "opencv_python",
        "yaml": "pyyaml",
    }

    roots = set()
    for text in sources:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])

    specs = sorted({
        allowed[aliases.get(r, r)]
        for r in roots
        if aliases.get(r, r) in allowed
    })

    if not specs:
        return ""

    return (
        'import subprocess,sys\n'
        'subprocess.run([sys.executable,"-m","pip","install",'
        '"--disable-pip-version-check","--quiet",*'
        + repr(specs)
        + '],check=True,timeout=120)\n'
    )


def tests_for_apps(input_output):
    tests = (
        json.loads(input_output)
        if isinstance(input_output, str) else input_output
    )
    if not isinstance(tests, dict) or not tests.get("inputs"):
        return None

    cases = list(zip(tests["inputs"], tests["outputs"]))

    return '''
import pytest
import subprocess
import sys

_CASES = %r
_NAME = %r

@pytest.mark.parametrize("index", range(len(_CASES)))
def test_apps(index):
    inp, expected = _CASES[index]
    if _NAME:
        scope = {"__name__": "candidate_module"}
        exec(compile(_SOURCE, "candidate.py", "exec"), scope)
        obj = scope.get(_NAME)
        if obj is None:
            obj = getattr(scope["Solution"](), _NAME)
        args = inp if isinstance(inp, list) else [inp]
        got = obj(*args)
        assert got == expected
    else:
        proc = subprocess.run(
            [sys.executable, "-c", _SOURCE],
            input=str(inp),
            text=True,
            capture_output=True,
            timeout=15,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == str(expected).strip()
''' % (cases, tests.get("fn_name"))


def verify(code, test_code=None, apps_io=None):
    from custome.podman_generating import run_pytest

    test_code = test_code or (
        tests_for_apps(apps_io) if apps_io else None
    )

    if not test_code:
        return {
            "passed": False,
            "status": "no_tests",
            "error_message": "No test cases provided",
            "error_types": ["NoTests"],
            "summary": {},
            "test_code": None,
            "coverage": None,
        }

    if apps_io:
        body = f"_SOURCE={code!r}\n" + test_code
    else:
        body = (
            f'exec(compile({code!r},"candidate.py","exec"),globals())\n'
            + test_code
        )

    script = bootstrap(code, test_code) + body

    try:
        report, stdout, stderr, exit_code = run_pytest(
            script, timeout=300
        )

        summary = (report or {}).get("summary", {})
        failures = []

        for test in (report or {}).get("tests", []):
            if test.get("outcome") != "passed":
                failures.append(str(
                    test.get("call", {}).get("longrepr")
                    or test.get("setup", {}).get("longrepr")
                    or test
                ))

        for collector in (report or {}).get("collectors", []):
            if collector.get("outcome") == "failed":
                failures.append(str(
                    collector.get("longrepr", collector)
                ))

        passed = (
            exit_code == 0
            and summary.get("passed", 0) > 0
            and summary.get("failed", 0) == 0
            and summary.get("error", 0) == 0
            and summary.get("errors", 0) == 0
        )

        message = "\n\n".join(failures) if failures else (
            stderr or stdout if not passed else ""
        )

        error_types = sorted(set(re.findall(
            r"\b[A-Za-z_]+(?:Error|Exception|Failure)\b",
            message,
        )))

        if not passed and not error_types:
            error_types = ["PytestFailure"]

        coverage = None
        if passed and os.environ.get("RL_CHECK_COVERAGE") == "1":
            try:
                from custome.podman_generating import run_test_coverage
                cov = run_test_coverage(script, timeout=300)
                coverage = (cov[0] or {}).get("totals")
            except Exception as exc:
                coverage = {"error": str(exc)}

        status = (
            "passed" if passed
            else "environment_error" if any(
                s in message for s in (
                    "ModuleNotFoundError", "No module named",
                    "pip install", "CalledProcessError",
                )
            )
            else "failed"
        )

        return {
            "passed": passed,
            "status": status,
            "error_message": message,
            "error_types": error_types,
            "summary": summary,
            "pytest_exit_code": exit_code,
            "test_code": test_code,
            "coverage": coverage,
        }

    except Exception as exc:
        return {
            "passed": False,
            "status": "environment_error",
            "error_message": str(exc),
            "error_types": [type(exc).__name__],
            "summary": {},
            "pytest_exit_code": None,
            "test_code": test_code,
            "coverage": None,
        }
