"""Phase 0 HF access check (CPU, ~1 min). Prints only status and file count, never the token.

Needs the owner to switch on the HF_TOKEN secret for THIS notebook (Add-ons > Secrets),
then "Save Version > Save & Run All".
"""
import json
from pathlib import Path

REPO = "Joe0204/dataset-ixi"
REV = "94862e0d6ce1234b1a63a49ac2b9af04020fbdeb"
result = {"repo": REPO, "revision": REV}

token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
    result["secret_attached"] = bool(token)
except Exception as e:  # noqa: BLE001
    result["secret_attached"] = False
    result["secret_error"] = type(e).__name__

from huggingface_hub import HfApi  # noqa: E402

api = HfApi()
for repo_type in ("dataset", "model"):
    try:
        files = api.list_repo_files(REPO, repo_type=repo_type, revision=REV, token=token or False)
        result[f"{repo_type}_status"] = "ok"
        result[f"{repo_type}_n_files"] = len(files)
        result[f"{repo_type}_files"] = files[:20]
        try:
            info = api.get_paths_info(REPO, files[:20], repo_type=repo_type, revision=REV, token=token or False)
            result[f"{repo_type}_sizes"] = {p.path: getattr(p, "size", None) for p in info}
        except Exception as e:  # noqa: BLE001
            result[f"{repo_type}_sizes_error"] = type(e).__name__
        break
    except Exception as e:  # noqa: BLE001
        code = getattr(getattr(e, "response", None), "status_code", None)
        result[f"{repo_type}_status"] = f"{type(e).__name__} http={code}"
del token

Path("/kaggle/working/hf_check.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
