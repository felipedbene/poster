#!/usr/bin/env bash
# Usage: ./set_github_secrets.sh
# Requires: GitHub CLI (gh) authenticated and a `.env` file in project root.

set -o allexport
source .env
set +o allexport

# Loop over variables in .env and set them as GitHub secrets
for var in $(grep -v '^#' .env | sed 's/=.*//'); do
  echo "Setting secret $var"
  gh secret set $var --body "${!var}"
done