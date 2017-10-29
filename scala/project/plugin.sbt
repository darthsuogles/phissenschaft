
resolvers ++= Seq(
  Resolver.url(
    "artifactory", url("http://scalasbt.artifactoryonline.com/scalasbt/sbt-plugin-releases")
  )(Resolver.ivyStylePatterns),
  "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/",
  "sonatype-releases" at "https://oss.sonatype.org/content/repositories/releases/",
  "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"
)

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "latest.integration")
addSbtPlugin("com.jsuereth" % "sbt-pgp" % "latest.integration")
addSbtPlugin("org.scoverage" % "sbt-scoverage" % "latest.integration")
addSbtPlugin("org.scalastyle" %% "scalastyle-sbt-plugin" % "latest.integration")
addSbtPlugin("com.github.gseitz" % "sbt-release" % "latest.integration")
addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "latest.integration")

addSbtPlugin("org.spark-packages" % "sbt-spark-package" % "0.2.6")
addSbtPlugin("org.foundweekends" % "sbt-bintray" % "0.5.1")
