package y.phi9t.core

import java.nio.file.{Path, Paths, Files}

case class FPath(jfp: Path) {
  import FPath.implicits._
  def /(sub: FPath) = new FPath(jfp.resolve(sub.jfp))
  def /(sub: String) = new FPath(jfp.resolve(Paths.get(sub)))
  override def toString(): String = jfp.toString
}

object FPath {
  object implicits {
    implicit def path2fpath(p: Path): FPath = new FPath(p)
    implicit def fpath2path(fp: FPath): Path = fp.jfp
    implicit def str2fpath(s: String): FPath = new FPath(Paths.get(s))
    implicit def fpath2str(fp: FPath): String = fp.jfp.toString
  }

  import implicits._
  lazy val cwd: FPath = Paths.get(System.getProperty("user.dir"))
  lazy val home: FPath = Paths.get(System.getProperty("user.home"))
}
