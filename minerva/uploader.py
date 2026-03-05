import asyncio
import hashlib
from pathlib import Path
from typing import Any, Callable

import httpx

from minerva.auth import auth_headers
from minerva.cache import job_cache
from minerva.constants import (
    JOB_STATUS_RETRIES,
    UPLOAD_CHUNK_RETRIES,
    UPLOAD_CHUNK_SIZE,
    UPLOAD_FINISH_RETRIES,
    UPLOAD_START_RETRIES,
)
from minerva.error_handling import _raise_if_upgrade_required, _retry_sleep, _retryable_status


async def get_session_status(
    upload_server_url: str, file_id: str, session_id: str, timeout: httpx.Timeout, headers: dict[str, Any]
) -> str:
    """
    status_data["state"] can be any of the following:
    - missing: The session_id does not exist on the server, likely expired. Need to start a new session.
    - failed: The upload session encountered an error during processing.
    - staging: The /upload/{file_id}/start was called and the session_id is valid, but no chunks have been uploaded yet.
    - queued: The file has been fully uploaded, but the server has not yet processed it.
    - uploading: The file is being transferred to the final storage backend.
    - uploaded: The file has been fully uploaded, transferred to final storage backend, and processed.
    """
    if not session_id:
        return "missing"

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, JOB_STATUS_RETRIES + 1):
            resp = await client.get(
                url=f"{upload_server_url}/api/upload/{file_id}/status",
                params={"session_id": session_id},
                headers=headers,
            )
            _raise_if_upgrade_required(resp)
            try:
                status_data = resp.json()
                return status_data.get("state", "missing")
            except Exception:
                pass
            if attempt != JOB_STATUS_RETRIES:
                await asyncio.sleep(_retry_sleep(attempt) / 2.5)
        return "missing"


async def upload_file(
    upload_server_url: str,
    token: str,
    path: Path,
    job: dict[str, Any],
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    # Fresh client per upload: multipart state must not be shared across coroutines
    headers = auth_headers(token)
    timeout = httpx.Timeout(connect=30, read=300, write=300, pool=30)
    file_id = job["file_id"]

    session_id = job.get("session_id")
    if session_id:
        session_status = await get_session_status(upload_server_url, file_id, session_id, timeout, headers)
        if session_status != "staging":
            # the session_id is no longer valid
            session_id = None
            job["session_id"] = None
            job_cache.set(job)
        if session_status in ("queued", "uploading", "uploaded"):
            # the file has already been uploaded
            job_cache.remove(job)
            raise RuntimeError("file already uploaded, skipping...")

    async with httpx.AsyncClient(timeout=timeout) as client:
        # 1. Start session
        if not session_id:
            for attempt in range(1, UPLOAD_START_RETRIES + 1):
                try:
                    resp = await client.post(f"{upload_server_url}/api/upload/{file_id}/start", headers=headers)
                    _raise_if_upgrade_required(resp)
                    if _retryable_status(resp.status_code):
                        if attempt == UPLOAD_START_RETRIES:
                            raise RuntimeError(f"upload start failed ({resp.status_code})")
                        await asyncio.sleep(_retry_sleep(attempt))
                        continue
                    resp.raise_for_status()
                    session_id = resp.json().get("session_id")
                    if not session_id:
                        raise RuntimeError("server did not return a session id")
                    job["session_id"] = session_id
                    job_cache.set(job)
                    break
                except httpx.HTTPError as e:
                    if attempt == UPLOAD_START_RETRIES:
                        raise RuntimeError(f"upload start failed ({e})") from e
                    await asyncio.sleep(_retry_sleep(attempt))
        if not session_id:
            raise RuntimeError("Failed to create upload session")

        # 2. Send chunks
        file_size = path.stat().st_size
        sent = 0
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                data = f.read(UPLOAD_CHUNK_SIZE)
                if not data:
                    break
                hasher.update(data)
                for attempt in range(1, UPLOAD_CHUNK_RETRIES + 1):
                    try:
                        resp = await client.post(
                            f"{upload_server_url}/api/upload/{file_id}/chunk",
                            params={"session_id": session_id},
                            headers={**headers, "Content-Type": "application/octet-stream"},
                            content=data,
                        )
                        _raise_if_upgrade_required(resp)
                        if _retryable_status(resp.status_code):
                            if attempt == UPLOAD_CHUNK_RETRIES:
                                raise RuntimeError(f"upload chunk failed ({resp.status_code})")
                            await asyncio.sleep(_retry_sleep(attempt, cap=20.0))
                            continue
                        resp.raise_for_status()
                        break
                    except httpx.HTTPError as e:
                        if attempt == UPLOAD_CHUNK_RETRIES:
                            raise RuntimeError(f"upload chunk failed ({e})") from e
                        await asyncio.sleep(_retry_sleep(attempt, cap=20.0))
                sent += len(data)
                if on_progress is not None:
                    on_progress(sent, file_size)

        # 3. Finish
        expected_sha256 = hasher.hexdigest()
        for attempt in range(1, UPLOAD_FINISH_RETRIES + 1):
            try:
                resp = await client.post(
                    f"{upload_server_url}/api/upload/{file_id}/finish",
                    params={"session_id": session_id, "expected_sha256": expected_sha256},
                    headers=headers,
                )
                _raise_if_upgrade_required(resp)
                if _retryable_status(resp.status_code):
                    if attempt == UPLOAD_FINISH_RETRIES:
                        raise RuntimeError(f"upload finish failed ({resp.status_code})")
                    await asyncio.sleep(_retry_sleep(attempt, cap=20.0))
                    continue
                resp.raise_for_status()
                result = resp.json()
                job_cache.remove(job)
                break
            except httpx.HTTPError as e:
                if attempt == UPLOAD_FINISH_RETRIES:
                    raise RuntimeError(f"upload finish failed ({e})") from e
                await asyncio.sleep(_retry_sleep(attempt, cap=20.0))

        return result


__all__ = ["upload_file"]
