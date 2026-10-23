"""Upload a GitHub release to Zenodo (new record, or new version of the existing record).

Called by .github/workflows/zenodo-release.yml. Metadata come from .zenodo.json.
The record is matched to earlier versions through the repository URL in related_identifiers.
"""
import datetime
import json
import os
import sys

import requests

TOKEN = os.environ.get("ZENODO_TOKEN", "")
SANDBOX = os.environ.get("ZENODO_SANDBOX", "").strip().lower() == "true"
BASE = "https://sandbox.zenodo.org/api" if SANDBOX else "https://zenodo.org/api"
REPO = os.environ["GITHUB_REPOSITORY"]
REPO_URL = f"https://github.com/{REPO}"
NAME = REPO.split("/", 1)[1]
TAG = os.environ["TAG"]
ARCHIVE = sys.argv[1]
H = {"Authorization": f"Bearer {TOKEN}"}


def fail(msg):
    print(f"::error::{msg}")
    sys.exit(1)


def call(method, url, **kw):
    r = requests.request(method, url, headers=H, timeout=300, **kw)
    if r.status_code >= 400:
        fail(f"Zenodo {method} {url} -> {r.status_code}: {r.text[:800]}")
    return r.json() if r.content else {}


if not TOKEN:
    fail("The secret ZENODO_TOKEN is missing (Organization settings > Secrets and variables > Actions).")

# ---- metadata -------------------------------------------------------------
if not os.path.exists(".zenodo.json"):
    fail(".zenodo.json is missing. Copy it from Tum-Soil-Biophysics/project-template and fill it in.")
with open(".zenodo.json", encoding="utf-8") as fh:
    meta = json.load(fh)

PLACEHOLDERS = ("[Repository name]", "[short description]", "[Family name]", "[Given name]", "[What the code", "[keyword]")
text = json.dumps(meta)
if any(p in text for p in PLACEHOLDERS):
    fail(".zenodo.json still contains placeholders like [Family name]. Fill in title, creators, description and keywords.")
if not meta.get("creators"):
    fail(".zenodo.json needs at least one creator.")

meta.setdefault("upload_type", "software")
meta.setdefault("title", NAME)
meta.setdefault("description", os.environ.get("RELEASE_BODY") or f"Release {TAG} of {REPO_URL}")
meta["version"] = TAG
meta["publication_date"] = datetime.date.today().isoformat()
related = [r for r in meta.get("related_identifiers", []) if r.get("identifier") not in (REPO_URL, f"{REPO_URL}/tree/{TAG}")]
related += [
    {"identifier": REPO_URL, "relation": "isSupplementTo", "scheme": "url"},
    {"identifier": f"{REPO_URL}/tree/{TAG}", "relation": "isIdenticalTo", "scheme": "url"},
]
meta["related_identifiers"] = related

# ---- find an earlier version of this repository --------------------------
previous = None
page = 1
while previous is None:
    batch = call("GET", f"{BASE}/deposit/depositions", params={"size": 100, "page": page, "sort": "mostrecent"})
    for dep in batch:
        ids = [r.get("identifier") for r in dep.get("metadata", {}).get("related_identifiers", [])]
        if dep.get("submitted") and REPO_URL in ids:
            previous = dep
            break
    if len(batch) < 100:
        break
    page += 1

# ---- create the draft -----------------------------------------------------
if previous:
    print(f"Earlier version found (record {previous['id']}); creating a new version.")
    meta.pop("communities", None)  # new versions inherit the communities of the first version
    nv = call("POST", f"{BASE}/deposit/depositions/{previous['id']}/actions/newversion")
    draft = call("GET", nv["links"]["latest_draft"])
    for f in draft.get("files", []):
        call("DELETE", f"{draft['links']['self']}/files/{f['id']}")
else:
    print("No earlier version found; creating a new record.")
    draft = call("POST", f"{BASE}/deposit/depositions", json={})

try:
    with open(ARCHIVE, "rb") as fh:
        call("PUT", f"{draft['links']['bucket']}/{os.path.basename(ARCHIVE)}", data=fh)
    call("PUT", draft["links"]["self"], json={"metadata": meta})
    done = call("POST", draft["links"]["publish"])
except SystemExit:
    requests.delete(draft["links"]["self"], headers=H, timeout=60)  # remove the unpublished draft
    raise

doi, concept = done["doi"], done.get("conceptdoi", "")
print(f"Published: https://doi.org/{doi}  (all versions: https://doi.org/{concept})")
with open(os.environ["GITHUB_OUTPUT"], "a") as out:
    out.write(f"doi={doi}\nconcept_doi={concept}\n")
with open(os.environ.get("GITHUB_STEP_SUMMARY", os.devnull), "a") as s:
    s.write(f"### Zenodo{' (sandbox)' if SANDBOX else ''}\n- This version: https://doi.org/{doi}\n- All versions: https://doi.org/{concept}\n")
