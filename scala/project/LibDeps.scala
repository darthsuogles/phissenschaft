package y.phi9t.sbt

import sbt._

object LibVer {
  lazy val scala = "2.11.11"
  lazy val spark = "2.2.0"
  lazy val ammonite = "1.0.0"
  lazy val scalameta = "1.7.0"
  lazy val akka = "2.4.17"
}

object LibDeps {

  //resolvers += "Typesafe Releases" at "http://repo.typesafe.com/typesafe/releases/"
  lazy val akka = Seq(
    "akka-actor",
    "akka-agent",
    "akka-cluster",
    "akka-cluster-metrics",
    "akka-cluster-sharding",
    "akka-cluster-tools",
    "akka-stream",
    "akka-slf4j",
    "akka-testkit"
  ) map { "com.typesafe.akka" %% _ % LibVer.akka }

  lazy val ammonite = Seq(
    "com.lihaoyi" % s"ammonite_${LibVer.scala}" % LibVer.ammonite,
    "org.scalameta" %% "scalameta" % LibVer.scalameta
  )
}
