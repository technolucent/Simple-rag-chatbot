#!/bin/bash
echo "Cleaning pip packages..."
pip freeze | xargs pip uninstall -y 2>/dev/null
echo "Cleaning caches..."
rm -rf ~/.cache/huggingface ~/.cache/pip
echo "Done."
