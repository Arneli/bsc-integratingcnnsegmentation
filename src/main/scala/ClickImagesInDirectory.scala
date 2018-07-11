import java.io.File

import scalismo.faces.gui.controllers.LMClickerViewController
import utils.ScallopOptions

import scala.util.Try

/**
  * Created by Marco on 29.08.2016.
  */



object ClickImagesInDirectory {

  def main(args: Array[String]): Unit = {
    val opts = new ProgramOptions(args)
    opts.verify()

    val stdDir = new File(opts.dir())
    println(s"Image Directory: $stdDir")

    // note: the whish was to have rw permissions for all users, so we adapt the save method
    LMClickerViewController(stdDir)
    println("done")
  }

  /** Object holds manipulation configuration for one id */
  class ProgramOptions(args: Seq[String]) extends ScallopOptions(args) {
    version(versionAndCopyright)

    banner(
      """|
         |Options:""".stripMargin)

    footer(
      """|
         |Example usage:
         |java -Xmx4g -cp scimap.jar scripts.ClickAll""".stripMargin)

    val dir = opt[String]("dir", default = Some("images"), descr = "directory for which to click images")

  }

}
