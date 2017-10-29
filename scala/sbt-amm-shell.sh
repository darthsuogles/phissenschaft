#!/bin/bash

set -eu

OS="$(uname -s | tr [:upper:] [:lower:])"
if [[ "darwin" == "${OS}" ]]; then
    which greadlink &>/dev/null || brew install coreutils
    function tracelink { greadlink -f $@; }
else
    function tracelink { readlink -f $@; }
fi

_bsd_="$(cd "$(dirname "$(tracelink "${BASH_SOURCE[0]}")")" && pwd)"

JAVA_OPTS="-Xmx4G -Xms32M"
_jvm_agent="${_bsd_}/agent/.agents/mem-inst.jar"
_opt_interp="${INTERP:-mod}"

function java_exec { java ${JAVA_OPTS} -javaagent:"${_jvm_agent}" $@ ; }

# TODO: Ammonite does not pass instrument libraries to the REPL
# Using the default Ammonite REPL 
function repl_amm {
    java_exec \
        -cp "$(cat "${_bsd_}/repl/SBT_RUNTIME_CLASSPATH")" \
        ammonite.Main \
        $@
}

# Using our customized Ammonite based REPL
function repl_mod {
    java_exec \
         -cp "$(cat "${_bsd_}/repl/SBT_RUNTIME_CLASSPATH")" \
         y.phi9t.repl.ReplMain \
         $@
}

# Using standard Scala REPL
function repl_scala {
    java_exec \
        -Dscala.usejavacp=true \
        -cp "$(cat "${_bsd_}/repl/SBT_RUNTIME_CLASSPATH")" \
        scala.tools.nsc.MainGenericRunner \
        $@
}

case "${_opt_interp}" in 
    mod) repl_mod ;;
    amm) repl_amm ;;
    scala) repl_scala ;;
esac
