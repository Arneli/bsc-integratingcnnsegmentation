package utils

import org.rogach.scallop.ScallopConf
import org.rogach.scallop.exceptions.ScallopException

class ScallopOptions(args: Seq[String]) extends ScallopConf(args) {

  val versionAndCopyright = "version 0.1 (c) University of Basel"

  override def onError(e: Throwable) = e match {
    case ScallopException(message) =>
      printHelp
      println("args → "+args.mkString(" "))
      println(message)
      sys.exit(128)
    case ex => super.onError(ex)
  }

}