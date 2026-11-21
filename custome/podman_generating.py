import json
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

REMOTE_URL="http://127.0.0.1:18765"

def remote_execute(
    method,
    code, 
    timeout=60,
    parse_json=False,
):
    token=os.environ["SYNTHGEN_REMOTE_TOKEN"]
    
    payload=json.dumps({
        "method" : method,
        "code" : code,
        "timeout" : timeout,
        "parse_json" : parse_json,
    }).encode("utf-8")

    request = Request(
        url=REMOTE_URL + "/execute",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    
    try:
        with urlopen(
            request,
            timeout=max(900, timeout + 300),
        ) as response:
            result=json.load(response)
    except HTTPError as e:
        details=e.read().decode(
            "utf-8", errors="replace"
        )
        raise RuntimeError(
            f"Remote Podman failed: {details}"
        ) from e
    
    except URLError as e:
        raise RuntimeError(
            f"Cannot connect to local Podman: {e}"
        ) from e
    return tuple(result["result"])

def execute_python(
    code, 
    timeout=60,
    parse_json=False,
):
    return remote_execute(
        "execute_python",
        code,
        timeout,
        parse_json,
    )

def run_pytest(code, timeout=60):
    return remote_execute(
        "run_pytest",
        code,
        timeout,
    )

def run_test_coverage(code,timeout=60):
    return remote_execute(
        "run_test_coverage",
        code,
        timeout,
    )