#!/bin/bash

# # get rustup
# # don't use hombrew
# curl https://sh.rustup.rs -sSf | sh

rustup self update
# get nightly compiler
rustup update nightly

# after nightly installed
rustup component add rls-preview --toolchain nightly
rustup component add rust-analysis --toolchain nightly
rustup component add rust-src --toolchain nightly
