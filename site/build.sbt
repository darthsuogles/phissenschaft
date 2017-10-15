version := "0.1.0"

scalaVersion := "2.11.11"

licenses += ("Apache-2.0", url("http://opensource.org/licenses/Apache-2.0"))

// http://www.scala-sbt.org/sbt-site/generators/sphinx.html
enablePlugins(GhpagesPlugin, SphinxPlugin)

git.remoteRepo := "git@github.com:apache-spark-on-k8s/userdocs.git"
