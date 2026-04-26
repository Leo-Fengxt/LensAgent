"""Google Drive uploader for the lensing orchestrator.

Handles authentication, folder creation, and file uploads to a structured
Google Drive hierarchy:

    lensing-final/
        <campaign>/
            task_000/
                afms/    (summary, best_params, best_fit, best_single)
                prl/     (best_params, best_fit, best_single)
                rsi/     (summary, best_params, pull map, best_single)
            task_018/
                ...

All operations are retry-safe and non-fatal: upload failures are logged
but never crash the orchestrator.  Token refresh is automatic.
"""

import logging
import mimetypes
import os
import time
from pathlib import Path
from threading import Lock
from typing import Optional

log = logging.getLogger("lensagent.drive")

LENSING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_PATH = os.path.join(LENSING_DIR, "token.json")
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
ROOT_FOLDER_NAME = "lensing-final"

MAX_RETRIES = 3
RETRY_BACKOFF = 5

_AFMS_UPLOAD_GLOBS = [
    "results_bundle/summary.txt",
    "results_bundle/best/best_params.json",
    "results_bundle/best/best_fit.png",
    "results_bundle/best/best_single.png",
    "results_bundle/best/repro_manifest.json",
    "results_bundle/best/repro_arrays.npz",
    "results_bundle/best/repro_arrays_map.json",
    "results_bundle/best/observation_bundle.pkl",
]

_PRL_UPLOAD_GLOBS = [
    "prl/best_params.json",
    "prl/best_fit.png",
    "prl/best_iter_0999.png",
    "prl/best_single.png",
    "prl/repro_manifest.json",
    "prl/repro_arrays.npz",
    "prl/repro_arrays_map.json",
    "prl/observation_bundle.pkl",
]

_RSI_UPLOAD_GLOBS = [
    "summary.txt",
    "best_params_chi2.json",
    "best_params_phys.json",
    "pull_map.png",
    "pull_map_candidates.png",
    "best_iter_0999.png",
    "best_single.png",
    "rsi_results.zip",
    "repro_manifest.json",
    "repro_arrays.npz",
    "repro_arrays_map.json",
    "observation_bundle.pkl",
]


class DriveUploader:
    """Thread-safe Google Drive uploader with lazy initialization."""

    def __init__(self, token_path: str = TOKEN_PATH, enabled: bool = True):
        self._token_path = token_path
        self._enabled = enabled
        self._service = None
        self._lock = Lock()
        self._folder_cache: dict[str, str] = {}

        if not enabled:
            log.info("Drive upload disabled")
            return
        if not os.path.exists(token_path):
            log.warning("token.json not found at %s — Drive upload disabled",
                        token_path)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_service(self):
        """Lazy-init and return the Drive API service, refreshing if needed."""
        if self._service is not None:
            return self._service

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials.from_authorized_user_file(self._token_path, SCOPES)
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                Path(self._token_path).write_text(creds.to_json())
                log.info("Refreshed Drive token")
            else:
                log.error("Drive token invalid and cannot refresh. "
                          "Re-run auth_once.py.")
                self._enabled = False
                return None

        self._service = build("drive", "v3", credentials=creds,
                              cache_discovery=False)
        return self._service

    def _find_or_create_folder(self, name: str,
                               parent_id: Optional[str] = None) -> Optional[str]:
        """Find existing folder by name under parent, or create it."""
        cache_key = f"{parent_id or 'root'}:{name}"
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        svc = self._get_service()
        if svc is None:
            return None

        q = (f"name='{name}' and mimeType='application/vnd.google-apps.folder' "
             f"and trashed=false")
        if parent_id:
            q += f" and '{parent_id}' in parents"

        resp = svc.files().list(q=q, spaces="drive",
                                fields="files(id,name)").execute()
        files = resp.get("files", [])
        if files:
            fid = files[0]["id"]
            self._folder_cache[cache_key] = fid
            return fid

        meta = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            meta["parents"] = [parent_id]

        folder = svc.files().create(body=meta, fields="id").execute()
        fid = folder["id"]
        self._folder_cache[cache_key] = fid
        log.info("Created Drive folder: %s (id=%s)", name, fid[:12])
        return fid

    def _ensure_path(self, *parts: str) -> Optional[str]:
        """Ensure a nested folder path exists, return the leaf folder ID."""
        parent = None
        for part in parts:
            parent = self._find_or_create_folder(part, parent)
            if parent is None:
                return None
        return parent

    def _upload_file(self, local_path: str, folder_id: str,
                     drive_name: Optional[str] = None) -> bool:
        """Upload a single file. Returns True on success."""
        svc = self._get_service()
        if svc is None:
            return False

        local = Path(local_path)
        if not local.exists():
            return False

        name = drive_name or local.name
        mime = mimetypes.guess_type(local.name)[0] or "application/octet-stream"

        from googleapiclient.http import MediaFileUpload
        media = MediaFileUpload(str(local), mimetype=mime, resumable=True)
        meta = {"name": name, "parents": [folder_id]}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                svc.files().create(
                    body=meta, media_body=media,
                    fields="id,name").execute()
                return True
            except Exception as exc:
                log.warning("Drive upload attempt %d/%d for %s: %s",
                            attempt, MAX_RETRIES, name, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)
                    svc = self._get_service()
                    if svc is None:
                        return False
        return False

    def upload_afms(self, task_id: int, afms_dir: str,
                    campaign_name: str) -> None:
        """Upload AFMS key outputs after AFMS completes."""
        if not self._enabled:
            return
        with self._lock:
            try:
                folder = self._ensure_path(
                    ROOT_FOLDER_NAME, campaign_name,
                    f"task_{task_id:03d}", "afms")
                if folder is None:
                    return

                count = 0
                for rel in _AFMS_UPLOAD_GLOBS:
                    full = os.path.join(afms_dir, rel)
                    if os.path.exists(full):
                        if self._upload_file(full, folder):
                            count += 1

                log.info("task %03d: uploaded %d AFMS files to Drive",
                         task_id, count)
            except Exception as exc:
                log.error("task %03d: Drive AFMS upload error: %s",
                          task_id, exc)

    def upload_prl(self, task_id: int, afms_dir: str,
                   campaign_name: str) -> None:
        """Upload PRL key outputs after AFMS completes."""
        if not self._enabled:
            return
        with self._lock:
            try:
                folder = self._ensure_path(
                    ROOT_FOLDER_NAME, campaign_name,
                    f"task_{task_id:03d}", "prl")
                if folder is None:
                    return

                count = 0
                for rel in _PRL_UPLOAD_GLOBS:
                    full = os.path.join(afms_dir, rel)
                    if os.path.exists(full):
                        if self._upload_file(full, folder):
                            count += 1

                log.info("task %03d: uploaded %d PRL files to Drive",
                         task_id, count)
            except Exception as exc:
                log.error("task %03d: Drive PRL upload error: %s",
                          task_id, exc)

    def upload_rsi(self, task_id: int, rsi_dir: str,
                   campaign_name: str) -> None:
        """Upload RSI key outputs after RSI completes."""
        if not self._enabled:
            return
        with self._lock:
            try:
                folder = self._ensure_path(
                    ROOT_FOLDER_NAME, campaign_name,
                    f"task_{task_id:03d}", "rsi")
                if folder is None:
                    return

                count = 0
                for rel in _RSI_UPLOAD_GLOBS:
                    full = os.path.join(rsi_dir, rel)
                    if os.path.exists(full):
                        if self._upload_file(full, folder):
                            count += 1

                log.info("task %03d: uploaded %d RSI files to Drive",
                         task_id, count)
            except Exception as exc:
                log.error("task %03d: Drive RSI upload error: %s",
                          task_id, exc)

    def upload_campaign_summary(self, campaign_dir: str,
                                campaign_name: str) -> None:
        """Upload the final campaign.json at the end."""
        if not self._enabled:
            return
        with self._lock:
            try:
                folder = self._ensure_path(ROOT_FOLDER_NAME, campaign_name)
                if folder is None:
                    return
                cj = os.path.join(campaign_dir, "campaign.json")
                if os.path.exists(cj):
                    self._upload_file(cj, folder)
                    log.info("Uploaded campaign.json to Drive")
                ol = os.path.join(campaign_dir, "orchestrator.log")
                if os.path.exists(ol):
                    self._upload_file(ol, folder)
                    log.info("Uploaded orchestrator.log to Drive")
            except Exception as exc:
                log.error("Drive campaign summary upload error: %s", exc)
