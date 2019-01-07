#!/bin/bash

set -eu -o pipefail

OS="$(uname -s | tr [:upper:] [:lower:])"
case "${OS}" in
    darwin)
        which greadlink &>/dev/null || brew install coreutils
        function tracelink { greadlink -f $@; }
        ;;
    linux)
        function tracelink { readlink -f $@; }
        ;;
    \?) >&2 echo "ERROR: unknown OS ${OPTARG}"
        exit 1
        ;;
esac

_bsd_="$(cd "$(dirname "$(tracelink "${BASH_SOURCE[0]}")")" && pwd)"

JAVA_OPTS="-Xmx20G -Xms512M"
_jvm_agent="${_bsd_}/agent/.agents/mem-inst.jar"
_opt_interp="${INTERP:-spark}"

function java_exec {
    java ${JAVA_OPTS} \
         -javaagent:"${_jvm_agent}" \
         -Djava.library.path="${_bsd_}/native_lib" \
         $@
}

_SPARK_JAVA_CLASSPATH="${_bsd_}/spark-3.0.0-SNAPSHOT-bin-master-build/jars/*"

# Using our customized Ammonite based REPL
function repl_mod {
    java_exec \
         -cp "$(cat "${_bsd_}/repl/.sbt.classpath/SBT_RUNTIME_CLASSPATH")" \
         y.phi9t.repl.ReplMain \
         $@
}

# Using our customized Ammonite based REPL + Spark
function repl_spark_mod {
    java_exec \
         -cp "${_SPARK_JAVA_CLASSPATH}:$(cat "${_bsd_}/.spark.repl/.sbt.classpath/SBT_RUNTIME_CLASSPATH")" \
         y.phi9t.repl.ReplMain \
         $@
}

# Using standard Scala REPL
function repl_scala {
    java_exec \
        -Dscala.usejavacp=true \
        -cp "$(cat "${_bsd_}/repl/.sbt.classpath/SBT_RUNTIME_CLASSPATH")" \
        scala.tools.nsc.MainGenericRunner \
        $@
}

case "${_opt_interp}" in
    mod) repl_mod ;;
    spark) repl_spark_mod ;;
    scala) repl_scala ;;
    \?) >&2 echo "unknown REPL type ${_opt_interp}"
esac
