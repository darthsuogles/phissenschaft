
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

// Build a runnable script with all dependencies
lazy val genClasspath = taskKey[Unit]("Build runnable script with classpath")

lazy val stdSettings = commonSettings ++ Seq(
  genClasspath := {
    import java.io.PrintWriter
    // Find all jar dependencies and construct a classpath string
    val fout = new PrintWriter((baseDirectory.value / "SBT_RUNTIME_CLASSPATH").toString)
    println("Building runtime classpath for current project")
    val cp: Classpath = (fullClasspath in Runtime).value
    fout.write(cp.files.map(_.toString).mkString(":"))
    fout.close()
  }
)

lazy val repl = (project in file("repl")).
  settings(stdSettings: _*).
  settings(
    name := "repl",
    libraryDependencies ++= akka ++ Seq(
      "com.lihaoyi" % s"ammonite_${scalaVer}" % "0.8.2"
    )
  )

lazy val root = (project in file(".")).
  settings(stdSettings: _*).
  settings(
    name := "template",
    libraryDependencies ++= akka ++ Seq(
      "com.lihaoyi" % s"ammonite_${scalaVer}" % "0.8.2"
    )
  )
