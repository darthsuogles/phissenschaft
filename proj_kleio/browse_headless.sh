#!/bin/bash

chrome='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

#    --remote-debugging-port=9222 \

"${chrome}" \
    --headless \
    --disable-gpu \
    --screenshot \
    https://www.chromestatus.com   # URL to open. Defaults to about:blank.
