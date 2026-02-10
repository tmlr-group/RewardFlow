#!/usr/bin/env bash

ENV_NAME="${1:-alfworld}"

if [[ "$ENV_NAME" == "alfworld" ]]; then
  echo "Launching AlfWorld agent..."
  python3 -m examples.prompt_agent.gpt4o_alfworld
elif [[ "$ENV_NAME" == "appworld" ]]; then
  echo "Launching AppWorld agent..."
  python3 -m examples.prompt_agent.gpt4o_appworld
else
  echo "Error: Unsupported environment '$ENV_NAME'. Use 'alfworld' or 'appworld'." >&2
  exit 1
fi
