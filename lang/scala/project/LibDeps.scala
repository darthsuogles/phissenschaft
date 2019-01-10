package y.phi9t.sbt

import sbt._

object LibVer {
  lazy val scala = scala_2_12
  lazy val scala_2_12 = "2.12.8"

  lazy val ammonite = "1.5.0"

  lazy val scalameta = "3.6.0"
  lazy val scalameta_2_12 = "3.6.0"

  // Spark
  lazy val spark = "2.4.0"
  lazy val sparkMaster = "3.0.0-SNAPSHOT"

  // TensorFlow
  lazy val tensorflowMaster = "0.4.2-SNAPSHOT"

  // Akka: https://github.com/akka/akka/releases
  lazy val akka = "2.5.19"
  // BEAM via Scio
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

  private lazy val sparkModules = Seq(
    "spark-core",
    "spark-sql",
    "spark-streaming",
    "spark-mllib",
    "spark-graphx"
  )

  lazy val sparkMaster = sparkModules.map {
    "org.apache.spark" %% _ % LibVer.sparkMaster }

  lazy val spark = sparkModules.map {
    "org.apache.spark" %% _ % LibVer.spark }

  lazy val tensorflowMaster = Seq(
    "org.platanios" %% "tensorflow" % LibVer.tensorflowMaster
  )

  lazy val ammonite = Seq(
    // Ammonite needs full scala version to match
    "com.lihaoyi" % s"ammonite_${LibVer.scala}" % LibVer.ammonite,
    "org.scalameta" %% "scalameta" % LibVer.scalameta
  )
}
