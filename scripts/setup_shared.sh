#!/usr/bin/env bash
set -euo pipefail

# Shared storage configuration
SHARED_DIR="/shared"
TEAM_GROUP="mlteam"

# Require administrator privileges
if [[ "$EUID" -ne 0 ]]; then
    echo "Please run this script with sudo."
    exit 1
fi

# Create the shared Linux group
groupadd -f "$TEAM_GROUP"

# Create shared directories
mkdir -p "$SHARED_DIR"/{models,data,checkpoints}

# Configure group ownership and permissions
chgrp -R "$TEAM_GROUP" "$SHARED_DIR"
chmod -R g+rwX,o-rwx "$SHARED_DIR"
find "$SHARED_DIR" -type d -exec chmod g+s {} +

# Install ACL tools if needed
if ! command -v setfacl >/dev/null 2>&1; then
    apt-get update
    apt-get install -y acl
fi

# Configure permissions for future files
find "$SHARED_DIR" -type d -exec \
    setfacl -d -m g:"$TEAM_GROUP":rwx {} +

# Add supplied Linux usernames to the group
for username in "$@"; do
    if id "$username" >/dev/null 2>&1; then
        gpasswd -a "$username" "$TEAM_GROUP"
    else
        echo "User not found: $username"
    fi
done

echo "Shared directory setup complete."
echo "Group members must reconnect and verify access."
ls -ld "$SHARED_DIR" "$SHARED_DIR"/*
