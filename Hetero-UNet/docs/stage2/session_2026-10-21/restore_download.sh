#!/usr/bin/env bash
set -euo pipefail
mkdir -p /hy-tmp/stage2_restore_archives
cd /hy-tmp/stage2_restore_archives
for object in Dataset701_AbdomenCT-20260910T013250Z-1-001.zip Dataset701_AbdomenCT-20260910T013250Z-1-002.zip Dataset701_AbdomenCT-20260910T013250Z-1-003.zip 3d_fullres_weights.tar; do
  printf '\nDOWNLOADING %s UTC=%s\n' "$object" "$(date -u +%FT%TZ)"
  oss cp "oss://datasets/$object" .
done
oss cp oss://backup/pred50_artifacts_20261013.zip .
printf '\nDOWNLOADS_COMPLETE UTC=%s\n' "$(date -u +%FT%TZ)"
sha256sum ./*.zip ./*.tar > archive_sha256.txt
du -h ./*
