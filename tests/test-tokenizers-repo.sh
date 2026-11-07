#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 2 ]; then
    printf "Usage: $0 <git-repo> <target-folder> [<test-exe>]\n"
    exit 1
fi

if [ $# -eq 3 ]; then
    toktest=$3
else
    toktest="./test-tokenizer-0"
fi

if [ ! -x "$toktest" ]; then
    printf "Test executable \"$toktest\" not found!\n"
    exit 1
fi

repo=$1
folder=$2
revision=${LLAMA_TEST_VOCABS_REVISION:-}
if [ -n "$revision" ] && [[ ! $revision =~ ^[0-9a-f]{40}$ ]]; then
    printf 'Vocabulary revision must be a full commit SHA.\n' >&2
    exit 1
fi

if [ -d "$folder" ] && [ -d "$folder/.git" ]; then
    if [ -z "$revision" ]; then
        git -C "$folder" pull --ff-only
    fi
else
    git clone -- "$repo" "$folder"

    # byteswap models if on big endian
    if [ "$(uname -m)" = s390x ]; then
        for f in "$folder"/*/*.gguf; do
            echo YES | python3 "$(dirname "$0")/../gguf-py/gguf/scripts/gguf_convert_endian.py" "$f" big
        done
    fi
fi

if [ -n "$revision" ]; then
    git -C "$folder" fetch origin "$revision"
    git -C "$folder" checkout --detach "$revision"
    git -C "$folder" lfs pull
fi

shopt -s globstar nullglob
result=0
executed=0
for gguf in "$folder"/**/*.gguf; do
    if [ -f "$gguf.inp" ] && [ -f "$gguf.out" ]; then
        executed=$((executed + 1))
        "$toktest" "$gguf" || result=1
    else
        printf "Found \"$gguf\" without matching inp/out files, ignoring...\n"
    fi
done

if [ "$executed" -eq 0 ]; then
    printf 'No complete vocabulary fixtures were tested.\n' >&2
    exit 1
fi
exit "$result"

