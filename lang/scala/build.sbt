import y.phi9t.sbt.{LibDeps, LibVer}

lazy val commonSettings = Seq(
  organization := "y.phi9t",
  version := "0.1",
  scalaVersion := LibVer.scala,
  scalacOptions ++= scalafixScalacOptions.value ++ Seq(
    "-Ywarn-unused-import"
  ),
  resolvers ++= Seq(
    DefaultMavenRepository,
    Resolver.mavenLocal,
    Resolver.file("local", file(Path.userHome.absolutePath + "/.ivy2/local"))(Resolver.ivyStylePatterns)
  )
)

// Runing a task on the aggregate project will also run it on the aggregated projects.
// For details, please read multi-part projects
// http://www.scala-sbt.org/0.13/docs/Multi-Project.html
lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "root",
    publishArtifact := false
  ).aggregate(
    /* Core modules */
    core,
    repl,
    agent,
    /* REPL: Apache Spark */
    sparkRepl,
    /* REPL: Google Beam */
    //scioRepl,
  )

lazy val core = (project in file("core")).
  settings(commonSettings: _*).
  settings(
    name := "core",
    publishArtifact := false
  )

lazy val ammonite = (project in file("ammonite"))

lazy val repl = (project in file("repl")).
  settings(commonSettings: _*).
  settings(
    name := "repl",
    scalaVersion := LibVer.scala,
    libraryDependencies ++=
      LibDeps.ammonite ++ LibDeps.akka ++ LibDeps.tensorflowMaster
  ).aggregate(agent).aggregate(core).dependsOn(core)

lazy val agent = (project in file("agent")).
  settings(commonSettings: _*).
  settings(
    assemblyJarName in assembly := "mem-inst.jar",
    packageOptions in (Compile, packageBin) +=
      Package.ManifestAttributes(
        "Premain-Class" -> "y.phi9t.instrument.ObjectSizeFetcher"),
    assemblyOutputPath in assembly := {
      val outFP = baseDirectory.value / ".agents" / (assemblyJarName in assembly).value
      println(outFP)
      outFP
    }
  )

// REPL: Apache Spark
lazy val sparkRepl = (project in file(".spark.repl"))
  .settings(commonSettings: _*)
  .settings(
    name := "sparkRepl",
    libraryDependencies ++= LibDeps.spark
  ).aggregate(repl).dependsOn(repl)
