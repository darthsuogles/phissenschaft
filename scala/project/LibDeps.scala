package y.phi9t.sbt

import sbt._

object LibVer {
  lazy val scala = "2.11.11"
  lazy val spark = "2.2.1"
  lazy val sparkMaster = "2.4.0-SNAPSHOT"
  lazy val ammonite = "1.0.3"
  lazy val scalameta = "2.1.2"
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

  lazy val spark = Seq(
    "spark-core",
    "spark-sql",
    "spark-streaming",
    "spark-mllib",
    "spark-graphx"
  ).map { "org.apache.spark" %% _ % LibVer.sparkMaster }

  lazy val ammonite = Seq(
    "com.lihaoyi" % s"ammonite_${LibVer.scala}" % LibVer.ammonite,
    "org.scalameta" %% "scalameta" % LibVer.scalameta
  )
}
