version := "0.1.0"

scalaVersion := "2.11.11"

licenses += ("Apache-2.0", url("http://opensource.org/licenses/Apache-2.0"))

// http://www.scala-sbt.org/sbt-site/generators/sphinx.html
enablePlugins(SphinxPlugin)

scmInfo := Some(ScmInfo(
  url("https://github.com/darthsuogles/spinnen-krawl"),
  "git@github.com:darthsuogles/spinnen-krawl.git"))
enablePlugins(GhpagesPlugin)
//git.remoteRepo := scmInfo.value.get.connection
git.remoteRepo := "git@github.com:darthsuogles/spinnen-krawl.git"
