# Task checkout setup for the swebench resources server (rows with a `setup_patch`, e.g. SWE-smith).
# Applies the row's setup patch to the clean checkout, removes the paths the agent must not see
# (HIDDEN_FILES; empty in the grading container, where the graded tests stay), and makes the
# result the repository's only commit: a root commit on branch _nel_task, so no parent, ref,
# stash or reflog entry leads back to the clean tree. anti_cheat_setup.sh, run afterwards in
# agent sandboxes, deletes every other ref and prunes the unreachable objects.
# Environment: WORKING_DIRECTORY (the repository), SETUP_PATCH (the uploaded diff),
# HIDDEN_FILES (newline-separated paths relative to the repository, may be empty).
_T0=$(date +%s 2>/dev/null || echo 0)
if ! cd "$WORKING_DIRECTORY" 2>/dev/null; then echo "SETUP_PATCH failed reason=no_workdir wd=$WORKING_DIRECTORY"; exit 2; fi
git config --global --add safe.directory "$WORKING_DIRECTORY" 2>/dev/null
if ! git rev-parse --git-dir >/dev/null 2>&1; then echo "SETUP_PATCH failed reason=not_a_git_repo wd=$WORKING_DIRECTORY"; exit 2; fi
if ! git reset -q --hard; then echo "SETUP_PATCH failed reason=reset_failed"; exit 2; fi
# SWE-smith's apply chain (swesmith.constants.GIT_APPLY_CMDS), resetting between attempts as its gather step does.
_APPLIED=
for _cmd in "git apply --verbose" "git apply --verbose --reject" "patch --batch --fuzz=5 -p1 -i"; do
    if [ -z "$_APPLIED" ]; then
        if $_cmd "$SETUP_PATCH"; then
            _APPLIED=1
        else
            git reset -q --hard
            git ls-files --others --exclude-standard -z -- '*.rej' '*.orig' | xargs -0 -r rm -f --
        fi
    fi
done
if [ -z "$_APPLIED" ]; then echo "SETUP_PATCH failed reason=apply_failed"; exit 3; fi
_HIDDEN=0
while IFS= read -r _path; do
    if [ -n "$_path" ] && [ -e "$_path" ]; then rm -rf -- "$_path"; _HIDDEN=$((_HIDDEN + 1)); fi
done <<EOF
$HIDDEN_FILES
EOF
if ! git add -A; then echo "SETUP_PATCH failed reason=add_failed"; exit 4; fi
if ! git checkout -q --orphan _nel_task; then echo "SETUP_PATCH failed reason=orphan_failed"; exit 4; fi
if ! git -c user.name=nemo-gym -c user.email=nemo-gym@localhost -c commit.gpgsign=false commit -q --no-verify --allow-empty -m "Initial commit"; then
    echo "SETUP_PATCH failed reason=commit_failed"; exit 4
fi
_ELAPSED=$(( $(date +%s 2>/dev/null || echo 0) - _T0 ))
echo "SETUP_PATCH applied hidden=$_HIDDEN head=$(git rev-parse HEAD) elapsed_s=$_ELAPSED"
