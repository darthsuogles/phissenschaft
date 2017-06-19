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
  ).dependsOn(agent)

lazy val agent = (project in file("agent")).
  settings(commonSettings: _*).
  settings(
    assemblyJarName in assembly := "mem-inst.jar",
    packageOptions in (Compile, packageBin) +=
      Package.ManifestAttributes( "Premain-Class" -> "ObjectSizeFetcher" ),
    assemblyOutputPath := {
      val outFP = baseDirectory.value / ".agents"
      println(outFP)
      outFP
    }
  )

lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "root"
  ).aggregate(repl)
