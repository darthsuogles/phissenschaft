/**
  * Build Ammonite type REPL with custom configurations
  */
package y.phi9t.repl

import java.io.{File, InputStream, OutputStream}

import ammonite.interp.Interpreter
import ammonite.ops._
import ammonite.runtime.{History, Storage}
import ammonite.{ Main => AmmReplMain }
import ammonite.main.Defaults
import ammonite.repl.{Repl, ReplApiImpl, SessionApiImpl}
import ammonite.util._
import ammonite.util.Util.newLine


object ReplMain extends App {
  val welcomeBanner = {
    def ammoniteVersion = ammonite.Constants.version
    def scalaVersion = scala.util.Properties.versionNumberString
    def javaVersion = System.getProperty("java.version")
    Util.normalizeNewlines(
      s"""phissenschaft (amm $ammoniteVersion)
          |(Scala $scalaVersion, Java $javaVersion)""".stripMargin
    )
  }

  val predef = """
  |repl.prompt() = "scala> ";
  |repl.frontEnd() = ammonite.repl.FrontEnd.JLineUnix;
  |repl.colors() = ammonite.util.Colors.BlackWhite
""".stripMargin

  val replMain = AmmReplMain(
    predef = predef,
    welcomeBanner = Some(welcomeBanner)
  )  

  replMain.run()
}
