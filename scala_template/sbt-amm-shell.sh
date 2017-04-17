#!/bin/bash

OS="$(uname -s | tr [:upper:] [:lower:])"
if [[ "darwin" == "${OS}" ]]; then
    which greadlink &>/dev/null || brew install coreutils
    function rlq { greadlink -f $@; }
else
    function rlq { readlink -f $@; }
fi

_bsd_="$(cd "$(dirname "$(rlq "${BASH_SOURCE[0]}")")" && pwd)"

# Using the default Ammonite REPL 
function repl_amm {
    java -cp "$(cat "${_bsd_}/SBT_RUNTIME_CLASSPATH")" \
         ammonite.Main \
         --predef='repl.prompt() = "scala> "; repl.frontEnd() = ammonite.repl.FrontEnd.JLineUnix; repl.colors() = ammonite.util.Colors.BlackWhite' \
         $@
}

# Using our customized Ammonite based REPL
function repl_mod {
    java -cp "$(cat "${_bsd_}/repl/SBT_RUNTIME_CLASSPATH")" y.phi9t.repl.ReplMain $@
}

repl_mod
