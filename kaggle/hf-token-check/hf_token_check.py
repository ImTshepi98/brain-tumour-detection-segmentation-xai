"""Phase 0 HF access check (CPU, ~1 min). Prints only status and file lists, never the token.

Needs the owner to switch on the HF_TOKEN secret for THIS notebook (Add-ons > Secrets),
then "Save Version > Save & Run All".

Repos checked:
  * Joe0204/dataset-ixi at the pinned revision (raw IXI zip, PLAN section A).
  * Joe0204/brats-3d-unet-checkpoint (holds ixi_preprocessed_sri24.zip used by the
    owner's September stage-1 notebook; provenance unknown, checked as a possible shortcut).
"""
import json
from pathlib import Path

REPOS = [("Joe0204/dataset-ixi", "94862e0d6ce1234b1a63a49ac2b9af04020fbdeb"),
         ("Joe0204/brats-3d-unet-checkpoint", None)]
result = {}

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
for repo, rev in REPOS:
    r = result.setdefault(repo, {"revision": rev})
    for repo_type in ("dataset", "model"):
        try:
            files = api.list_repo_files(repo, repo_type=repo_type, revision=rev, token=token or False)
            r["repo_type"] = repo_type
            r["status"] = "ok"
            r["n_files"] = len(files)
            r["files"] = files[:50]
            try:
                info = api.get_paths_info(repo, files[:50], repo_type=repo_type, revision=rev, token=token or False)
                r["sizes"] = {p.path: getattr(p, "size", None) for p in info}
            except Exception as e:  # noqa: BLE001
                r["sizes_error"] = type(e).__name__
            break
        except Exception as e:  # noqa: BLE001
            code = getattr(getattr(e, "response", None), "status_code", None)
            r[f"{repo_type}_status"] = f"{type(e).__name__} http={code}"
del token

Path("/kaggle/working/hf_check.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
