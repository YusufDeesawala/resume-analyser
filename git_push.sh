#!/bin/bash

# Set Git username and token for authentication
GITHUB_USERNAME="MUSTAFA892"
GITHUB_TOKEN="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"

# Ask for the branch name
read -p "Enter the branch name to push: " BRANCH_NAME

# Push the committed changes to the specified branch
git push https://$GITHUB_USERNAME:$GITHUB_TOKEN@$(git config --get remote.origin.url | sed 's/https:\/\///') $BRANCH_NAME

# Uncomment below lines if you want to manually specify the remote repository as well
# REMOTE_REPO="origin"   # Change if needed
# git push https://$GITHUB_USERNAME:$GITHUB_TOKEN@$(git remote get-url $REMOTE_REPO) $BRANCH_NAME
  