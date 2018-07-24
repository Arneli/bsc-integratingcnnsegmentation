import java.io.{BufferedWriter, File, FileWriter}

import api.Fitting
import scalismo.faces.color.{RGB, RGBA}
import scalismo.faces.image.PixelImageOperations
import scalismo.faces.io.{MoMoIO, PixelImageIO, RenderParameterIO, TLMSLandmarksIO}
import scalismo.faces.landmarks.TLMSLandmark2D
import scalismo.faces.sampling.face.MoMoRenderer
import utils.ScallopOptions
import java.lang.management.ManagementFactory
import java.lang.management.ThreadMXBean


object FitWithOcclusions extends App{



  override def main(args: Array[String]): Unit = {
    val timestamp_start_overall: Long = System.currentTimeMillis
    val timestamp_start_overall_CPU: Long = getCpuTime

    val opt = new ProgramOptions(args)
    opt.verify()
    println(
      s"""|Fitting with options ...
          |    ... model: ${opt.model()}
          |    ... fitting steps: ${opt.numSteps()}
          |    ... debug mode is: ${if(opt.debug())"on" else "off"}
          |    ... fitting method is: ${opt.fitMethod()}""".stripMargin)

    val posterior_logger_file = new File(s"posteriors_${opt.fitMethod()}.txt")
    val time_logger_file = new File(s"times_${opt.fitMethod()}.txt")
    val bw_pl = new BufferedWriter(new FileWriter(posterior_logger_file))
    val bw_tl = new BufferedWriter(new FileWriter(time_logger_file))

    fitSingle(opt.model(), opt.debug(), opt.numSteps(),opt.fitMethod(), bw_pl, bw_tl)

    val timestamp_end_overall: Long = System.currentTimeMillis
    val timestamp_end_overall_CPU: Long = getCpuTime
    var elapsed_time_overall:Float = (((timestamp_end_overall-timestamp_start_overall).toFloat/1000)*10).toInt/10F
    val elapsed_time_overall_CPU:Float = (((timestamp_end_overall_CPU-timestamp_start_overall_CPU).toFloat/1000000000)*10).toInt/10F
    bw_tl.write(s"\n\nFor all images in the 'data_in' directory, the fit took ${elapsed_time_overall} seconds (Wall-Clock Time)\nand ${elapsed_time_overall_CPU} seconds (CPU Time)")

    bw_pl.close()
    bw_tl.close()
    System.exit(0)
  }

  def fitSingle(model: String, debug: Boolean, fitSteps: Int = 1000, method:String, bw_pl:BufferedWriter, bw_tl:BufferedWriter, occlusion: Boolean = false): Unit ={
    for(i <- 0 to 9) {
    //(0 until 10).par.foreach( i =>{
    //try{
        // ATTENTION: When using "x until y" x is inclusive, y is exclusive
        val timestamp_start_wallClock: Long = System.currentTimeMillis
        val timestamp_start_CPU: Long = getCpuTime
        scalismo.initialize()
        val full_path_to_source_image = "data_in/test" + i + ".png"
        val full_path_to_source_LMfile = "data_in/test" + i + ".tlms"
        val targetFile = new File(full_path_to_source_image)
        val target = PixelImageIO.read[RGBA](targetFile).get
        val modelFace12 = MoMoIO.read(new File(model)).get

        println(s"start fitting target $target")
        val targetLMList = TLMSLandmarksIO.read2D(new File(full_path_to_source_LMfile)).get

        val rpsFile = new File("data_out", targetFile.getName.replace(".png", s"_${method}.rps"))
        val maskFile = new File("data_out", targetFile.getName.replace(".png", s"_mask_${method}_.png"))
        val fitFile = new File("data_out", targetFile.getName.replace(".png", s"_fit_${method}.png"))
        val overlayFile = new File("data_out", targetFile.getName.replace(".png", s"_overlay_${method}.png"))

        val (rps, mask) = Fitting.fitWithOcclusions(targetFile.getName, target, targetLMList, modelFace12, fitSteps, debug, Some(rpsFile.toString), Some(maskFile.toString), Some(overlayFile.toString), method, i, bw_pl)

        RenderParameterIO.write(rps, rpsFile).get
        PixelImageIO.write(mask.map(RGB(_)), maskFile)

        val renderingFit = MoMoRenderer(modelFace12).renderImage(rps)
        PixelImageIO.write(renderingFit, fitFile)

        val fitOverlay = PixelImageOperations.alphaBlending(target.map {
          _.toRGB
        }, renderingFit)
        PixelImageIO.write(fitOverlay, overlayFile)

        val timestamp_end_wallClock: Long = System.currentTimeMillis
        val elapsed_time_wallCLock: Float = (((timestamp_end_wallClock - timestamp_start_wallClock).toFloat / 1000) * 10).toInt / 10F
        val timestamp_end_CPU: Long = getCpuTime
        val elapsed_time_CPU: Float = (((timestamp_end_CPU - timestamp_start_CPU).toFloat / 1000000000) * 10).toInt / 10F
        bw_tl.write(s"test${i}_${method}_wallClock: ${elapsed_time_wallCLock}\ntest${i}_${method}_CPU: ${elapsed_time_CPU}\n")
      }
//      catch{
//        case e: Throwable =>
//          println(s"Something went wrong with test${i}")
//          println(s"${e.getMessage}")
//          println(s"${e.getStackTrace}")
//          e.printStackTrace()
//      }
//      })
    //}
  }

  def getCpuTime:Long = {
    val bean = ManagementFactory.getThreadMXBean
    if (bean.isCurrentThreadCpuTimeSupported) bean.getCurrentThreadCpuTime
    else 0.toLong
  }
}


/** Object holds fit configuration with id and number of fitting steps*/
class ProgramOptions(args: Seq[String]) extends ScallopOptions(args) {
  version(versionAndCopyright)

  banner(
    """|
       |Options:""".stripMargin)

  footer(
    """|
       |Example usage:
       |java -cp scimap.jar scripts.FitAll -i myID -n 10000 -d true
       |java -cp target/scala-2.12/bsc-integratingCNNSegmentation.jar FitWithOcclusions -d -l data/harry01.tlms -t data/harry01.png -m /mnt/kortad01/scala/parametric-face-image-generator/data/bfm2017/model2017-1_face12_nomouth.h5 -n 1000
       |java -cp target/scala-2.12/bsc-integratingCNNSegmentation.jar FitWithOcclusions -d -m bfm/model2017-1_face12_nomouth.h5 -n 1000 -f EGGER""".stripMargin)
  // Now, for the updated code, only the last line is relevant


  val model = opt[String]  ("model", required = true,       descr = "model to fit")
  val debug     = opt[Boolean] ("debug",     default = Some(false),  descr = "Shows status every 100 fit steps")
  val numSteps  = opt[Int]     ("num-steps", default = Some(10000), descr = "number of fitting steps")
  val fitMethod = opt[String] ("fit-method", required = true, descr = "Method for the fitting-process [EGGER|GROTRU|FCN|DUMMY]")
}
