import y.phi9t.sbt.{LibDeps, LibVer}

lazy val commonSettings = Seq(
  organization := "y.phi9t",
  version := "0.1",
  scalaVersion := LibVer.scala,
  resolvers ++= Seq(
    DefaultMavenRepository,
    Resolver.mavenLocal,
    Resolver.file("local", file(Path.userHome.absolutePath + "/.ivy2/local"))(Resolver.ivyStylePatterns)
  )
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

// TODO: check these `spPackage::artifactPath`
lazy val spkgs = (project in file(".spark-packages")).
  settings(
    scalaVersion := LibVer.scala,
    sparkVersion := LibVer.spark
  )
  .dependsOn(spkgAvro, spkgCoreNLP)
  .aggregate(spkgAvro, spkgCoreNLP)

lazy val spkgAvro = (project in file("spark-avro")).
  settings(
    scalaVersion := LibVer.scala,
    sparkVersion := LibVer.spark,
    name := "spark-avro",
    version := "0.3.0-edge-SNAPSHOT",
    spName := s"databricks/${name.value}"
  )

lazy val spkgCoreNLP = (project in file("spark-corenlp")).
  settings(
    scalaVersion := LibVer.scala,
    sparkVersion := LibVer.spark,
    sparkComponents ++= Seq(
      "sql", "core"
    ),
    libraryDependencies ++= Seq(
      "edu.stanford.nlp" % "stanford-corenlp" % "3.6.0",
      "com.google.protobuf" % "protobuf-java" % "2.6.1"
      //"edu.stanford.nlp" % "stanford-corenlp" % "3.6.0" % "test" classifier "models",
      //"org.scalatest" %% "scalatest" % "2.2.6" % "test"
    ),
    organization := "databricks",
    name := "spark-corenlp",
    version := "0.3.0-edge-SNAPSHOT",
    spName := s"databricks/${name.value}"
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
