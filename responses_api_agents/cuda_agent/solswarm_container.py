# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run SolSwarm's REAL agent container per rollout, instead of imitating it.

A Python imitation of ``docker/agent/entrypoint.sh`` (prompt renderer, skills
installer, env contract) would be a standing drift liability: when SolSwarm
changes how a sandbox is built, nothing tells us, and the model just trains
against a different world.

Instead, the published agent image is extracted once per node
(``ensure_enroot_base``) and each rollout runs inside a real instance of it
(``enroot start`` on the shared read-only rootfs, with only per-rollout
writable mounts — ``enroot_launch_argv``). SolSwarm's own ``entrypoint.sh``
then does the assembly itself: it installs the skills, stitches
``submission_rule.md``, renders the role prompt, writes the OpenCode config
and launches the agent. We supply only what the Kubernetes pod controller
would supply — environment variables, delivered as a sourced file
(``write_rollout_env``) because ``enroot start`` resets the calling
environment — and a platform endpoint to talk to. Model routing rides the
same contract: synthetic model-catalog and endpoint-registry files written
into the rollout tree (``write_registry_catalog_files``) plus the override
variables pointing the entrypoint at them (``registry_contract_env``), so the
entrypoint's own registered-endpoint dispatch configures OpenCode to dial the
per-rollout policy URL.

The rootfs is byte-for-byte the production image, so the residual difference
from production is the substrate (enroot on a compute node instead of
Kubernetes) — not package versions, sandbox structure, or assembly logic.

The file-listing script and its output live in the rollout directory, which is
mounted read-write into the agent's own container, so the IDENTICAL/DIFFERS
verdict is an integrity alarm against accident and drift, not a defense
against an agent that deliberately tampers with the listing.
"""

import fcntl
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

from responses_api_agents.cuda_agent import sandbox_manifest, solswarm_surface


LOG = logging.getLogger(__name__)


def prepare_rollout_dir(rollout_dir: Path, problem_dir: Path) -> None:
    """Create this rollout's private tree (see ROLLOUT_MOUNTS) around ``problem_dir``.

    ``problem_dir`` is moved in as ``<rollout_dir>/problem`` (or created empty
    when it does not exist), so callers seed the problem files first and then
    hand the directory over.
    """
    rollout_dir = Path(rollout_dir)
    for name, _ in ROLLOUT_MOUNTS:
        (rollout_dir / name).mkdir(parents=True, exist_ok=True)
    (rollout_dir / "sandbox_probe.py").write_text(sandbox_manifest.PROBE_SOURCE)
    target = rollout_dir / "problem"
    if not target.exists():
        if Path(problem_dir).is_dir():
            Path(problem_dir).rename(target)
        else:
            target.mkdir(parents=True, exist_ok=True)


def pod_identity_env(
    meta: dict[str, Any],
    rollout_id: str,
    language: str,
    platform_api_endpoint: str,
) -> dict[str, str]:
    """The rollout's platform endpoint and SolSwarm identity variables.

    Shared by both sandbox profiles: container mode receives them through
    ``build_env``, and the minimal profile with the /submit skill exports them
    directly into its subprocess environment. One definition keeps the two
    profiles' spellings from drifting.
    """
    return {
        "PLATFORM_API_ENDPOINT": platform_api_endpoint,
        "SOLSWARM_LANGUAGE": str(language or ""),
        # submit.py's precompliance gate refuses bundles in the wrong language.
        "SOLSWARM_REQUIRED_LANGUAGE": str(language or ""),
        # Identity (appears in traces/logs; also used for file naming). One
        # fixed campaign id: RL rollouts belong to no SolSwarm campaign.
        "SOLSWARM_CAMPAIGN_ID": "nemo-rl",
        # The prompt header renders these as distinct fields (Artifact ID / Agent
        # ID), so they must not be the same value: the artifact identifies the
        # PROBLEM (stable across the rollouts working on it, as in production
        # where it names the kernel artifact), the architect identifies THIS
        # agent instance.
        "SOLSWARM_ARTIFACT_ID": str((meta.get("definition") or {}).get("name") or "problem"),
        "SOLSWARM_ARCHITECT_ID": f"rl-{rollout_id[:8]}",
        "SOLSWARM_AGENT_RUN_NAME": f"nemo-rl-{rollout_id[:8]}",
        # Every rollout is its own round 1: there is no multi-round campaign.
        "SOLSWARM_ROUND_NUMBER": "1",
        "SOLSWARM_TRIAL_ROUND": "1",
    }


def build_env(
    *,
    meta: dict[str, Any],
    problem_dir: str,
    ban_framework_kernels: bool = True,
    ban_cuda_graphs: bool = True,
    opencode_config_extra: Optional[dict[str, Any]] = None,
    reasoning_variant: str = "",
    platform_api_endpoint: str,
    cudagym_url: str,
    rollout_id: str,
    max_duration: int,
) -> dict[str, str]:
    """The environment variables SolSwarm's pod controller would inject.

    This is the ONLY contract we own in container mode: what the sandbox looks
    like is then decided entirely by ``entrypoint.sh`` reading these variables.
    Model routing is deliberately absent here: ``registry_contract_env`` and
    the synthetic catalog files (``write_registry_catalog_files``) own it,
    including ``SOLSWARM_AGENT_MODEL`` and the endpoint credential.
    """
    env = {
        # Runtime + role dispatch.
        "AGENT_RUNTIME": "opencode",
        "SOLSWARM_AGENT_RUNTIME": "opencode",
        "AGENT_ROLE": "optimizer",
        # Problem + evaluation service.
        "PROBLEM_DIR": problem_dir,
        # SOLSWARM_PROBLEM is deliberately NOT set: resolve_problem_description()
        # returns early if we pre-set it, so setting it would suppress their own
        # derivation (definition.json .description, else "Optimize kernel: <name>")
        # and put a bare identifier in the prompt header where production shows a
        # human description.
        "CUDAGYM_URL": cudagym_url,
        # The production pod controller injects SOLSWARM_GPU_SPEC from the
        # campaign's resolved gpu_spec (campaign-controller agent_run.rs) and
        # the image currently ignores it; kept for controller parity.
        "SOLSWARM_GPU_SPEC": str(meta.get("target_hardware") or "b200").lower(),
        # The optimizer prompt renders ${SOLSWARM_TARGET_GPU:-...}. Production
        # exports it only on the entrypoint's database path, which a
        # pre-populated PROBLEM_DIR bypasses, so the same value (the lowercase
        # GPU class id) is derived here; no target_hardware leaves it unset and
        # the prompt renders its own default.
        "SOLSWARM_TARGET_GPU": str(meta.get("target_hardware") or "").lower(),
        # entrypoint.sh strips the ban sections from optimizer.md unless these
        # are exactly "true" (it sets them only on its database path), and
        # submit.py reads the same variables for its local compliance gate.
        # Banned LIBRARIES need no variable: they ride submission_rule.md.
        "SOLSWARM_BAN_FRAMEWORK_KERNELS": str(ban_framework_kernels).lower(),
        "SOLSWARM_FORBID_FRAMEWORK_KERNELS": str(ban_framework_kernels).lower(),
        "SOLSWARM_BAN_CUDA_GRAPHS": str(ban_cuda_graphs).lower(),
        # The entrypoint generates the OpenCode config itself, discarding ours;
        # it deep-merges this JSON fragment over the generated config as the
        # last step of its config assembly (see SOLSWARM_OPENCODE_CONFIG_EXTRA
        # there), so training-critical settings — compaction off, context
        # limit — survive. Compaction during training silently drops
        # post-compaction turns from the token-id trajectory.
        "SOLSWARM_OPENCODE_CONFIG_EXTRA": json.dumps(opencode_config_extra) if opencode_config_extra else "",
        # Effort/variant --variant flag. The entrypoint ALWAYS passes one:
        # an explicit value here overrides; "" (dropped by the empty-value
        # filter below) leaves the entrypoint's own default. The training vLLM
        # accepts the flag.
        "SOLSWARM_OPENCODE_VARIANT": reasoning_variant,
        "SOLSWARM_OPENCODE_REASONING_EFFORT": reasoning_variant,
        # The platform endpoint and identity variables, shared with the
        # minimal profile (pod_identity_env).
        **pod_identity_env(meta, rollout_id, str(meta.get("language") or ""), platform_api_endpoint),
        # Budget: the entrypoint wraps the agent in `timeout $SOLSWARM_MAX_DURATION`.
        "SOLSWARM_MAX_DURATION": str(max_duration),
        # Image-native locations.
        "SOLSWARM_TRACE_DIR": "/traces",
        "HOME": "/home/agent",
        # Keep OpenCode from replacing itself mid-rollout (the image pins it).
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "NO_UPDATE_NOTIFIER": "1",
    }
    return {k: v for k, v in env.items() if v not in (None, "")}


# Provider name the entrypoint's hub-catalog OpenCode config declares
# (entrypoint.sh, opencode_provider_for_model / configure_opencode). The merge
# fragment below re-keys per-model settings onto it.
# The entrypoint's registered-endpoint dispatch declares this provider (the
# prefix of its client model "solswarm-model-endpoint/<registry id>"), and
# keys the model under the registry id. The overlay fragment must use these
# generated names, not the agent config's own.
CONTAINER_OPENCODE_PROVIDER = "solswarm-model-endpoint"
REGISTRY_POLICY_ID = "rl-policy"


def container_opencode_config_extra(opencode_config: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate the agent's opencode_config into the entrypoint's merge fragment.

    Container mode discards our OpenCode config: the entrypoint generates its
    own and deep-merges ``SOLSWARM_OPENCODE_CONFIG_EXTRA`` over it last (a
    recursive jq object merge; the fragment wins on leaves). Only the
    training-critical settings travel in the fragment, and they must use names
    the GENERATED config declares, not ours:

      * ``compaction`` and ``permission`` are top-level, provider-independent
        keys and pass through unchanged, so they apply whatever provider the
        entrypoint declares.
      * The per-model ``limit`` block — the context-window half of the
        compaction mitigation — is looked up under the agent config's own
        ``<provider>/<model id>`` and re-keyed onto
        ``CONTAINER_OPENCODE_PROVIDER``/``REGISTRY_POLICY_ID``, the provider
        and model key the registered-endpoint dispatch actually generates; the
        agent config's names exist only in our discarded config, so an
        un-re-keyed limit could never match a generated entry.
      * ``small_model`` is re-qualified to those generated names for the same
        reason.
    """
    opencode_config = opencode_config or {}
    extra: dict[str, Any] = {k: v for k, v in opencode_config.items() if k in ("compaction", "permission")}
    provider_name, _, model_id = model.partition("/")
    model_id = model_id or model
    if opencode_config.get("small_model"):
        extra["small_model"] = f"{CONTAINER_OPENCODE_PROVIDER}/{REGISTRY_POLICY_ID}"
    limit = (
        (((opencode_config.get("provider") or {}).get(provider_name) or {}).get("models") or {}).get(model_id) or {}
    ).get("limit")
    if limit:
        extra["provider"] = {CONTAINER_OPENCODE_PROVIDER: {"models": {REGISTRY_POLICY_ID: {"limit": limit}}}}
    return extra


# In-container path the whole rollout dir is mounted at. Everything crossing
# the container boundary rides this one mount: the launch script sources the
# env contract from it, runs the file-listing script from it, and writes the
# start stamp and the entrypoint transcript back into it.
STAGE = "/mnt/solswarm-rollout"
# The rollout-dir subdir that becomes the entrypoint's /workspace.
WS_SUBDIR = "ws"
# Where the --rc launch script redirects the entrypoint's own stdout+stderr.
ENTRYPOINT_LOG = "entrypoint.log"
# First act of the in-container launch script: stamp the wall clock into the
# rollout dir. The startup latency is this stamp minus the wall clock the
# agent server read just before exec'ing the launcher.
CONTAINER_START_STAMP = ".sandbox_start_epoch"

# Pairs of (per-rollout private directory, image path it is mounted at). Each
# is a bind so concurrent rollouts sharing one base never collide: the entrypoint
# wipes and rewrites $HOME/.config/opencode/skills every boot, deletes
# /skills-library after installing, and writes control files into the trace dir.
ROLLOUT_MOUNTS = (
    (WS_SUBDIR, "/workspace"),
    ("home", "/home/agent"),
    ("traces", "/traces"),
    ("artifacts", "/artifacts"),
    ("skills-library", "/skills-library"),
    ("skills-library-compat", "/skills-library-compat"),
)


def _reference_inputs_mtime(root: Path) -> Optional[int]:
    """Newest mtime among the checkout files that feed the reference sandbox.

    Those are the submission-rule fragments (the only checkout content stitched
    into the reference's problem dir; the agent image supplies everything
    else). Returns None when the fragment directory is missing or empty.
    """
    try:
        mtimes = [p.stat().st_mtime for p in (Path(root) / solswarm_surface.SUBMISSION_FRAGMENTS_SUBDIR).glob("*.md")]
    except OSError:
        return None
    return int(max(mtimes)) if mtimes else None


def checkout_id(root: Path) -> str:
    """Identity of the pinned SolSwarm checkout (git sha, else the fragments' mtime).

    One of the reference cache key's two components (the other, ``cache_tag``,
    carries the agent-image identity): the checkout supplies the stitched
    submission-rule fragments — its only input to the reference — so bumping
    the submodule invalidates the baseline instead of silently reusing a stale
    one. A dirty checkout (uncommitted fragment edits) folds the fragments'
    newest mtime into the id — the sha alone would keep serving the pre-edit
    baseline.
    """
    fragments = str(solswarm_surface.SUBMISSION_FRAGMENTS_SUBDIR)
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            sha = out.stdout.strip()[:12]
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--", fragments],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                stamp = _reference_inputs_mtime(root)
                return f"{sha}-dirty{stamp if stamp is not None else os.getpid()}"
            return sha
    except Exception:  # noqa: BLE001 - fall through to the mtime fingerprint
        pass
    stamp = _reference_inputs_mtime(root)
    if stamp is not None:
        return f"mtime-{stamp}"
    # Deliberately unique-per-process rather than a shared "unknown" bucket:
    # a constant here would make every checkout share one cached baseline,
    # which is exactly the staleness this function exists to prevent.
    return f"unidentified-{os.getpid()}"


# Settle budget for the reference sandbox: long enough for OpenCode to start
# and install its provider plugin (npm state a real rollout also has), short
# enough that it does no work.
REFERENCE_SETTLE_SECONDS = 75


def reference_tree(
    root: Path,
    cache_dir: Path,
    problem_files: dict[str, str],
    *,
    argv_builder: Callable[[Path, bool], list[str]],
    cache_tag: str,
    timeout: int = 900,
    extra_env: Optional[dict[str, str]] = None,
    seed: Optional[Callable[[Path], None]] = None,
) -> Optional[dict]:
    """Build the BASELINE sandbox by running SolSwarm's entrypoint, and record its file listing.

    This is the "what does a SolSwarm sandbox look like" side of the parity
    check, and it is generated rather than declared: the image's own entrypoint
    assembles a sandbox from the same problem inputs with none of the RL wiring
    (no model endpoint, no platform). Comparing a training rollout against it
    is a diff of two observed trees, so an upstream layout change shows up as a
    difference in BOTH and cancels out, while an RL-introduced difference
    stands alone.

    ``argv_builder`` supplies the launcher (``(rollout_dir, probe) -> argv``;
    the caller passes ``enroot_launch_argv`` bound to the extracted base),
    ``cache_tag`` carries the agent-image identity (the extracted base's name,
    which hashes the image path, mtime, and size — the baseline comes from the
    image, so a new image must invalidate it just as a new checkout does),
    ``extra_env`` carries launcher-specific variables
    (``enroot_subprocess_env``), and ``seed`` fills the rollout dir the way the
    rollout's launcher would (the per-rollout image copies).

    Cached per checkout id and image identity; returns None if the reference
    could not be built (the caller then simply skips the equality check).
    """
    cache_dir = Path(cache_dir)
    checkout = checkout_id(root)
    cache = cache_dir / f"reference_tree_{checkout}_{cache_tag}.json"
    # A readable cached baseline short-circuits the build.
    if cache.is_file():
        try:
            return json.loads(cache.read_text())
        except Exception:  # noqa: BLE001 - a corrupt cache should just rebuild
            pass

    # Fresh scratch rollout tree, seeded with the same problem files a rollout gets.
    rollout_dir = cache_dir / f"reference_rollout_{checkout}"
    shutil.rmtree(rollout_dir, ignore_errors=True)
    prepare_rollout_dir(rollout_dir, rollout_dir / "problem")
    for name, content in problem_files.items():
        (rollout_dir / "problem" / name).write_text(content)
    if seed is not None:
        # Without this the enroot reference boots over EMPTY skills/home
        # mounts, the entrypoint installs nothing, and the completeness check
        # below rejects the tree every time.
        seed(rollout_dir)

    # The same env contract as a rollout, with every endpoint deliberately
    # unreachable, including the same registry routing: the synthetic catalog
    # names a policy endpoint the agent cannot reach, so OpenCode starts up
    # (installing its plugin) and then fails its first call — settle, not work.
    contract = {
        **build_env(
            meta={"definition": {"name": "reference"}, "language": "triton", "target_hardware": "B200"},
            problem_dir="/tmp/problem",
            platform_api_endpoint="http://127.0.0.1:1/api/v1",  # unreachable on purpose
            cudagym_url="http://127.0.0.1:1",
            rollout_id="reference",
            max_duration=REFERENCE_SETTLE_SECONDS,
        ),
        **registry_contract_env("reference-model", "reference"),
    }
    write_registry_catalog_files(rollout_dir, "reference-model", "http://127.0.0.1:1/v1")
    # The container only sees environment sourced from the rollout's env file
    # (enroot start resets the calling environment), so the contract travels
    # via the file alone. The launcher PROCESS gets exactly what a rollout's
    # launcher gets (``extra_env``, i.e. enroot_subprocess_env): inheriting
    # os.environ here would let stray host variables — ENROOT_* above all —
    # steer only the reference. None inherits, subprocess.run's default.
    write_rollout_env(rollout_dir, contract)
    env = dict(extra_env) if extra_env is not None else None
    # Run the entrypoint until it settles, then read the listing it left in the tree.
    try:
        argv = argv_builder(rollout_dir, True)
        subprocess.run(
            argv,
            env=env,
            capture_output=True,
            timeout=timeout,
        )
        probe = json.loads((rollout_dir / "sandbox_probe.json").read_text())
    except Exception as e:  # noqa: BLE001 - reference is diagnostics, never fatal
        LOG.warning("could not build the SolSwarm reference sandbox: %s", e)
        return None

    # A half-assembled reference would be cached forever and make every later
    # rollout read as drift, so require positive evidence that the entrypoint
    # actually did its work before trusting it as the baseline.
    tree = probe.get("tree") or []
    required = ("/home/agent/.config/opencode/skills/", "/prompts/", "/tmp/problem/definition.json")
    missing = [marker for marker in required if not any(marker in entry for entry in tree)]
    if missing or len(tree) < 100 or probe.get("truncated"):
        LOG.warning(
            "SolSwarm reference sandbox looks incomplete (%d paths, missing %s%s); not caching",
            len(tree),
            missing,
            ", TRUNCATED" if probe.get("truncated") else "",
        )
        # Keep the failed attempt's transcript for diagnosis rather than deleting it.
        return None

    # A complete baseline: cache it and drop the scratch tree.
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(probe))
    shutil.rmtree(rollout_dir, ignore_errors=True)
    return probe


# ---------------------------------------------------------------------------
# enroot launch machinery. Instead of binding the checkout's files over the
# training container's rootfs, the published agent image is extracted once per
# node and each rollout is entered through `enroot start` — the image supplies every
# binary, package, skill, and prompt, and only the per-rollout directories are
# mounted in. The per-rollout cost is `enroot start` latency; see
# CONTAINER_START_STAMP.
# ---------------------------------------------------------------------------


def enroot_paths(data_root: Path) -> dict[str, str]:
    """ENROOT_* path environment for the inner enroot, under one local root.

    The cluster's ``/etc/enroot/enroot.conf`` points these at host paths
    (``/raid``) that do not exist inside the training container, and a per-job
    root keeps concurrent jobs on one node from sharing extraction state.
    """
    data_root = Path(data_root)
    return {
        "ENROOT_RUNTIME_PATH": str(data_root / "runtime"),
        "ENROOT_CACHE_PATH": str(data_root / "cache"),
        "ENROOT_DATA_PATH": str(data_root / "data"),
        "ENROOT_TEMP_PATH": str(data_root / "tmp"),
    }


def enroot_subprocess_env(data_root: Path) -> dict[str, str]:
    """Environment for the ``enroot`` process itself (never the container's).

    Built from scratch rather than inherited, because the caller's environment
    changes what enroot does on the HOST side. The rollout env carries the pod
    contract's ``HOME=/home/agent``, and with the cluster default
    ``ENROOT_MOUNT_HOME yes`` enroot binds ``$HOME`` from the host — a path
    that only exists inside the image. The ``VISIBLE_DEVICES`` variables
    trigger the nvidia/mellanox start hooks, which need container CLIs the
    training image does not carry; a rollout container is CPU-only, like
    production's agent pods (``void`` is enroot's documented no-op value).
    ``ENROOT_ROOTFS_WRITABLE=n`` keeps the shared base read-only regardless of
    cluster config, which is what makes concurrent starts on one base safe;
    the mountpoints a read-only start needs are created by
    ``ensure_enroot_base``. The container's own environment comes from the
    image plus the sourced ``ROLLOUT_ENV_FILE``, not from here.
    """
    return {
        "PATH": os.environ.get("PATH", "/usr/sbin:/usr/bin:/sbin:/bin"),
        "HOME": os.environ.get("HOME", "/root"),
        "NVIDIA_VISIBLE_DEVICES": "void",
        "MELLANOX_VISIBLE_DEVICES": "void",
        "ENROOT_MOUNT_HOME": "n",
        "ENROOT_ROOTFS_WRITABLE": "n",
        **enroot_paths(data_root),
    }


# Ceiling for one agent-image extraction (a few GB unsquashed to local disk).
ENROOT_CREATE_TIMEOUT = 1800


def ensure_enroot_base(image: Path, data_root: Path) -> tuple[str, Path, Optional[float]]:
    """Extract the agent image once into a shared read-only enroot container.

    Returns ``(container name, rootfs path, extraction seconds)``; the seconds
    are None when an earlier call already extracted it. A file lock serializes
    concurrent callers so the multi-GB unsquash happens exactly once per node,
    and a completion marker distinguishes a finished extraction from one that
    died mid-unsquash. Every rollout then enters this one rootfs read-only via
    ``enroot start``; all writable paths are per-rollout mounts.
    """
    image = Path(image)
    # The name hashes the image's path, mtime, and size, so a new agent image
    # gets a fresh extraction instead of silently reusing the old rootfs.
    stat = image.stat()
    tag = hashlib.sha256(f"{image}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:8]
    name = f"cuda-agent-{tag}"
    data_root = Path(data_root)
    for sub in ("runtime", "cache", "data", "tmp"):
        (data_root / sub).mkdir(parents=True, exist_ok=True)
    rootfs = data_root / "data" / name
    done_marker = data_root / f"{name}.complete"
    with open(data_root / f"{name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if done_marker.is_file():
            return name, rootfs, None
        # A rootfs without the marker is a previous extraction that died
        # partway; `enroot create` refuses an existing directory, so clear it.
        if rootfs.exists():
            shutil.rmtree(rootfs)
        started = time.monotonic()
        proc = subprocess.run(
            ["enroot", "create", "--name", name, str(image)],
            env={**os.environ, **enroot_paths(data_root)},
            capture_output=True,
            text=True,
            timeout=ENROOT_CREATE_TIMEOUT,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"enroot create failed for {image} (rc={proc.returncode}): {proc.stderr[-2000:]}")
        # `enroot start` runs this rootfs read-only (enroot_subprocess_env), so
        # every fstab destination must already exist; the image itself lacks
        # the pod-provided ones. They are created once here, while the tree is
        # still writable.
        for dest in (STAGE, *(d for _, d in ROLLOUT_MOUNTS), "/tmp/problem"):
            (rootfs / dest.lstrip("/")).mkdir(parents=True, exist_ok=True)
        done_marker.touch()
        return name, rootfs, time.monotonic() - started


def seed_rollout_from_image(rollout_dir: Path, rootfs: Path) -> None:
    """Populate the per-rollout copies of image paths the entrypoint mutates.

    ``/skills-library`` and ``/skills-library-compat`` are wiped by the
    entrypoint after installing, and ``/home/agent`` accumulates agent state,
    so each rollout mounts its own copy (ROLLOUT_MOUNTS) — here filled from
    the extracted image itself, so the contents are the image's, not the
    checkout's.
    """
    rollout_dir = Path(rollout_dir)
    rootfs = Path(rootfs)
    for name, src_sub in (
        ("skills-library", "skills-library"),
        ("skills-library-compat", "skills-library-compat"),
        ("home", "home/agent"),
    ):
        src = rootfs / src_sub
        if src.is_dir():
            shutil.copytree(src, rollout_dir / name, dirs_exist_ok=True, symlinks=False)


# The pod-controller env contract rides a file on the stage mount, sourced as
# the in-container script's first act: `enroot start` RESETS the calling
# environment (it applies the image's own env config instead), so subprocess
# env alone never reaches the entrypoint. The file also keeps the values out
# of the argv (visible in /proc on the node) and gives the container exactly
# what a production pod gets — a clean image environment plus the injected
# variables, not the training job's environment.
ROLLOUT_ENV_FILE = ".rollout-env"

# The in-container launch script, passed to `enroot start --rc`. Without --rc,
# enroot runs the image's Docker ENTRYPOINT as the command script and our argv
# arrives as its positional arguments (the entrypoint's "direct invocation"
# branch), instead of as the process to run.
ROLLOUT_RC_FILE = ".launch-rc.sh"


def write_rollout_env(rollout_dir: Path, contract_env: dict[str, str]) -> None:
    """Write the pod-controller env contract for the enroot runtime to source.

    The contract carries the evaluation-service credentials, so the file is
    created private at open time (mode 0o600, which the process umask can only
    narrow further): a create-then-chmod sequence would briefly expose it with
    the process's default creation mode.
    """
    lines = [f"export {key}={shlex.quote(str(value))}" for key, value in sorted(contract_env.items())]
    path = Path(rollout_dir) / ROLLOUT_ENV_FILE
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write("\n".join(lines) + "\n")


# Synthetic-catalog registry routing: how container mode points the entrypoint
# at the per-rollout policy endpoint. These files sit at the rollout-dir root,
# so they ride the rollout-dir -> STAGE bind and need no separate --mount: a
# read-only rootfs only accepts binds onto mountpoints created when the base
# image was extracted, and STAGE is one. Their in-container paths are handed to
# the entrypoint through the override env vars.
REGISTRY_MODELS_FILE = "agent-model-catalog.json"
REGISTRY_ENDPOINTS_FILE = "agent-model-endpoints.json"
REGISTRY_MODELS_STAGE_PATH = f"{STAGE}/{REGISTRY_MODELS_FILE}"
REGISTRY_ENDPOINTS_STAGE_PATH = f"{STAGE}/{REGISTRY_ENDPOINTS_FILE}"


def registry_contract_env(policy_id: str, api_key: str) -> dict[str, str]:
    """Env contract additions that route the model through the entrypoint's registry.

    Points SolSwarm's entrypoint at the synthetic catalog files
    ``write_registry_catalog_files`` drops into the rollout tree:
    ``SOLSWARM_AGENT_MODEL`` becomes the synthetic policy id, the two
    ``*_FILE`` variables are the override paths (symmetric with the operator's
    ConfigMap remap), and ``SOLSWARM_AGENT_MODEL_API_KEY`` is the credential
    the entrypoint hands OpenCode for the policy endpoint.
    """
    return {
        "SOLSWARM_AGENT_MODEL": policy_id,
        "SOLSWARM_AGENT_MODEL_API_KEY": api_key,
        "SOLSWARM_AGENT_MODEL_CATALOG_FILE": REGISTRY_MODELS_STAGE_PATH,
        "SOLSWARM_AGENT_MODEL_ENDPOINTS_FILE": REGISTRY_ENDPOINTS_STAGE_PATH,
    }


def write_registry_catalog_files(rollout_dir: Path, policy_id: str, policy_url: str) -> None:
    """Write the synthetic models-catalog and endpoints-registry into the rollout tree.

    These are the operator-sanctioned override files the entrypoint reads through
    the variables ``registry_contract_env`` sets. They declare one synthetic
    model, ``policy_id``, whose models-catalog ``route.policy`` is
    ``registered_endpoint`` and whose endpoints entry carries ``agent_base_url =
    policy_url`` -- the per-rollout capture-prefixed model-server URL
    (``<model_server>/ng-rollout/<id>[/training-token-capture]/v1``). OpenCode
    speaks the OpenAI-compatible wire natively, so the entrypoint's
    registered-endpoint dispatch configures OpenCode's provider to dial
    ``agent_base_url`` as-is, with no proxy in between: the rollout path prefix
    reaches the model server intact and per-rollout token capture works
    unchanged.

    The wire ``model`` OpenCode sends is ``policy_id``; the model server
    overrides it with its own ``config.model`` before reaching vLLM
    (``vllm_model`` app.py), so the synthetic id needs no relation to the served
    model name.
    """
    rollout_dir = Path(rollout_dir)
    runtimes = ["claude", "codex", "opencode", "n3"]
    models = {
        "version": 1,
        "models": [
            {
                "id": policy_id,
                "name": policy_id,
                "reasoning": {"default": "high", "efforts": ["low", "medium", "high", "max"]},
                "route": {"endpoint": policy_id, "policy": "registered_endpoint", "upstream_model": policy_id},
                "runtimes": runtimes,
            }
        ],
    }
    endpoints = {
        "version": 1,
        "endpoints": [
            {
                "id": policy_id,
                "runtimes": runtimes,
                "upstream_model": policy_id,
                "base_url": policy_url,
                "agent_base_url": policy_url,
                "litellm_provider": "openai",
                "temperature": 1.0,
                "top_p": 1.0,
            }
        ],
    }
    (rollout_dir / REGISTRY_MODELS_FILE).write_text(json.dumps(models))
    (rollout_dir / REGISTRY_ENDPOINTS_FILE).write_text(json.dumps(endpoints))


def enroot_launch_argv(
    base_name: str,
    *,
    rollout_dir: Path,
    probe: bool = False,
    entrypoint_override: Optional[str] = None,
) -> list[str]:
    """``enroot start`` argv that runs one rollout inside the extracted image.

    The image supplies the whole root filesystem; only the per-rollout
    directories are mounted, onto the image paths named by ROLLOUT_MOUNTS,
    plus a private writable ``/tmp`` (a read-only rootfs leaves the image's
    ``/tmp`` immutable, and the entrypoint writes scratch files there —
    production pods get a writable ``/tmp`` from the container layer) and
    production's ``/tmp/problem``, mounted after ``/tmp`` so it lands inside
    the private one. ``--root`` is passed because the training job runs as
    root, and pyxis-style enroot does not honor the image's USER directive
    either. The rootfs itself stays read-only
    (``enroot_subprocess_env``), which is what makes concurrent starts on one
    shared base safe. Callers must have written the env contract with
    ``write_rollout_env`` first.
    """
    rollout_dir = Path(rollout_dir)
    # The problem bind's mountpoint must exist inside the private /tmp.
    (rollout_dir / "tmp" / "problem").mkdir(parents=True, exist_ok=True)
    script = (
        "set -e\n"
        f". {STAGE}/{ROLLOUT_ENV_FILE}\n"
        f"date +%s.%N > {STAGE}/{CONTAINER_START_STAMP}\n"
        "cd /workspace\n"
        f"bash /entrypoint.sh > {STAGE}/{ENTRYPOINT_LOG} 2>&1 & EP=$!\n"
        # A TERM to this shell (the agent server kills the launcher's whole
        # process group on timeout/cancel, and enroot forwards signals to the
        # container command) must take the backgrounded entrypoint down with
        # it instead of orphaning it. `wait` returns >128 when the trap
        # interrupts it, so wait again until the entrypoint has actually
        # exited (`kill -0` = still running) before the listing and the exit.
        "trap 'kill -TERM $EP 2>/dev/null || true' TERM\n"
        "if wait $EP; then rc=0; else rc=$?; fi\n"
        'while [ "$rc" -gt 128 ] && kill -0 $EP 2>/dev/null; do\n'
        "  if wait $EP; then rc=0; else rc=$?; fi\n"
        "done\n"
    )
    if probe:
        script += (
            f"timeout 120 python3 {STAGE}/sandbox_probe.py > {STAGE}/sandbox_probe.json "
            f"2>{STAGE}/sandbox_probe.err || true\n"
        )
    script += "exit $rc\n"
    # --rc: enroot reads this host-side file and runs it as the container's
    # command script, replacing the one synthesized from the image's Docker
    # ENTRYPOINT (which would otherwise receive our command as its arguments).
    # The script sticks to POSIX sh because the image's default shell runs it.
    rc_path = rollout_dir / ROLLOUT_RC_FILE
    rc_path.write_text("#!/bin/sh\n" + script)
    rc_path.chmod(0o700)
    argv = ["enroot", "start", "--root", "--rc", str(rc_path), "--mount", f"{rollout_dir}:{STAGE}"]
    for name, dest in ROLLOUT_MOUNTS:
        argv += ["--mount", f"{rollout_dir / name}:{dest}"]
    argv += ["--mount", f"{rollout_dir / 'tmp'}:/tmp"]
    argv += ["--mount", f"{rollout_dir / 'problem'}:/tmp/problem"]
    # Test/debug affordance: mount a modified entrypoint over the image's own so
    # a changed entrypoint.sh can be exercised without rebuilding the agent
    # image. /entrypoint.sh already exists in the image, so this file-over-file
    # bind is valid on the read-only rootfs. Normally unset.
    override = entrypoint_override or os.environ.get("CUDA_AGENT_ENTRYPOINT_OVERRIDE")
    if override and Path(override).is_file():
        argv += ["--mount", f"{override}:/entrypoint.sh"]
    argv += [base_name]
    return argv
