import y.phi9t.sbt.{LibDeps, LibVer}

lazy val commonSettings = Seq(
  organization := "y.phi9t",
  version := "0.1",
  scalaVersion := LibVer.scala
)

lazy val repl = (project in file("repl")).
  settings(commonSettings: _*).
  settings(
    name := "repl",
    scalaVersion := LibVer.scala,
    libraryDependencies ++= LibDeps.ammonite
  ).aggregate(agent)

lazy val agent = (project in file("agent")).
  settings(commonSettings: _*).
  settings(
    assemblyJarName in assembly := "mem-inst.jar",
    packageOptions in (Compile, packageBin) +=
      Package.ManifestAttributes("Premain-Class" -> "ObjectSizeFetcher"),
    assemblyOutputPath in assembly := {
      val outFP = baseDirectory.value / ".agents" / (assemblyJarName in assembly).value
      println(outFP)
      outFP
    }
  )

lazy val spAvro = (project in file("spark-avro")).
  settings(
    scalaVersion := LibVer.scala
  )

lazy val spCoreNLP = (project in file("spark-corenlp")).
  settings(
    scalaVersion := LibVer.scala
  )


// Runing a task on the aggregate project will also run it on the aggregated projects.
// For details, please read multi-part projects
// http://www.scala-sbt.org/0.13/docs/Multi-Project.html
lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "root",
    publishArtifact := false
  ).aggregate(repl, agent)
