#!/bin/bash

set -eux

origin_url="$(git remote get-url origin)"           # <2> Save the current remote for later

tmp_dir="/tmp/gh-pages/proj"
rm -fr "${tmp_dir}" && mkdir -p "${tmp_dir}" && pushd "${tmp_dir}"

git init
git remote add origin "${origin_url}"
#git push origin --delete gh-pages && git branch -D gh-pages || echo "no existing gh-pages"
git checkout --orphan gh-pages
git rm -rf . || echo "no file at all"

git commit --allow-empty -m "re-init gh-pages"
git push --force origin gh-pages      # <4> Publish the repo's master branch as gh-pages

popd
