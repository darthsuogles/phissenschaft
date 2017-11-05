addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "latest.integration")

sbtPlugin := true

name := "sbt-spark-package"

organization := "org.spark-packages"

version := "0.2.7-SNAPSHOT"

description := "sbt plugin to develop, use, and publish Spark Packages"

libraryDependencies += "org.scalaj" %% "scalaj-http" % "latest.integration"

libraryDependencies += "org.apache.directory.studio" % "org.apache.commons.codec" % "1.8"

licenses := Seq("Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0"))

publishTo := Some("Spark Package Main Repo" at
    s"https://api.bintray.com/content/spark-packages/maven/spark-packages_sbt-spark-package/${version.value}")
