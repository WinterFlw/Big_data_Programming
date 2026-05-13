#!/bin/zsh

set -e

# Install a local Git commit-msg hook that enforces the v2 commit policy.
# Hooks live under .git/hooks and are not versioned, so this script makes the
# setup repeatable for every teammate.

repo_root="$(git rev-parse --show-toplevel)"
hook_path="$repo_root/.git/hooks/commit-msg"
validator="$repo_root/v2/scripts/validate_commit_message.py"
template="$repo_root/v2/.gitmessage"

if [[ ! -f "$validator" ]]; then
  echo "Validator not found: $validator"
  exit 1
fi

cat > "$hook_path" <<'EOF'
#!/bin/sh
repo_root="$(git rev-parse --show-toplevel)"
python3 "$repo_root/v2/scripts/validate_commit_message.py" "$1"
EOF

chmod +x "$hook_path"
git config commit.template "$template"

echo "Installed commit-msg hook: $hook_path"
echo "Configured commit template: $template"

