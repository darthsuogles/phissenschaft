{
  /*
  val pluginVersion = System.getProperty("plugin.version")
  if(pluginVersion == null)
    throw new RuntimeException("""|The system property 'plugin.version' is not defined.
                                 |Specify this property using the scriptedLaunchOpts -D.""".stripMargin)
  else
  */
  addSbtPlugin("org.spark-packages" % "sbt-spark-package" % "0.2.6-SNAPSHOT")
}
