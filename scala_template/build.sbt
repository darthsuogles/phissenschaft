
lazy val scalaVer = "2.12.1"

lazy val commonSettings = Seq(
  organization := "y.phi9t",
  version := "0.1",
  scalaVersion := scalaVer
)

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
) map { "com.typesafe.akka" %% _ % "2.4.17" }

lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "template",
    libraryDependencies ++= akka ++ Seq(
      "com.lihaoyi" % s"ammonite_${scalaVer}" % "0.8.2"
    )
  )
