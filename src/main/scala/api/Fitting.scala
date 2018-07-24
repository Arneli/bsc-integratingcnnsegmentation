package api

import java.io.{BufferedWriter, File, PrintStream}

import breeze.linalg.max
import scalismo.faces.color._
import scalismo.faces.deluminate.SphericalHarmonicsOptimizer
import scalismo.faces.image._
import scalismo.faces.io.{PixelImageIO, RenderParameterIO}
import scalismo.faces.landmarks._
import scalismo.faces.mesh._
import scalismo.faces.momo._
import scalismo.faces.parameters._
import scalismo.faces.sampling.face._
import scalismo.faces.sampling.face.evaluators.PixelEvaluators.IsotropicGaussianPixelEvaluator
import scalismo.faces.sampling.face.evaluators.PriorEvaluators._
import scalismo.faces.sampling.face.evaluators.{LabeledIndependentPixelEvaluator, _}
import scalismo.faces.sampling.face.loggers.{PrintLogger, VerbosePrintLogger}
import scalismo.faces.sampling.face.proposals.SphericalHarmonicsLightProposals._
import scalismo.faces.sampling.face.proposals._
import scalismo.geometry._
import scalismo.sampling.algorithms._
import scalismo.sampling.evaluators._
import scalismo.sampling.loggers._
import scalismo.sampling.proposals._
import scalismo.sampling.{DistributionEvaluator, MarkovChain, ProposalGenerator, TransitionRatio}
import scalismo.utils.Random

import scala.math.exp

/**
  * API-object provides functions for fitting
  */
object Fitting {
  def fitWithOcclusions(filename: String, targetImg : PixelImage[RGBA], targetLMList: IndexedSeq[TLMSLandmark2D], model: MoMo, fitSteps: Int, debug: Boolean = false, rpsFile: Option[String] = None, maskFile: Option[String] = None, overlayFile: Option[String] = None, method:String, b:Int, bw_pl:BufferedWriter): (RenderParameter,PixelImage[Int]) = {

    import scalismo.faces.color.{RGB, RGBA}
    import scalismo.faces.deluminate.SphericalHarmonicsOptimizer
    import scalismo.faces.image.{AccessMode, PixelImage}
    import scalismo.faces.parameters.RenderParameter
    import scalismo.faces.sampling.face.evaluators.PixelEvaluators._
    import scalismo.faces.sampling.face.evaluators.PointEvaluators.IsotropicGaussianPointEvaluator
    import scalismo.faces.sampling.face.evaluators.PriorEvaluators.{GaussianShapePrior, GaussianTexturePrior}
    import scalismo.faces.sampling.face.evaluators._
    //import scalismo.faces.sampling.face.loggers.PrintLogger
    import scalismo.faces.sampling.face.proposals.ImageCenteredProposal.implicits._
    import scalismo.faces.sampling.face.proposals.ParameterProposals.implicits._
    import scalismo.faces.sampling.face.proposals.SphericalHarmonicsLightProposals.{RobustSHLightSolverProposalWithLabel, _}
    import scalismo.faces.sampling.face.proposals._
    import scalismo.faces.sampling.face.{MoMoRenderer, ParametricImageRenderer, ParametricLandmarksRenderer, ParametricModel}
    import scalismo.faces.segmentation.LoopyBPSegmentation
    import scalismo.faces.segmentation.LoopyBPSegmentation.{BinaryLabelDistribution, Label, LabelDistribution}
    import scalismo.geometry.{Vector, Vector3D, _2D}
    import scalismo.sampling._
    import scalismo.sampling.algorithms.MetropolisHastings
    import scalismo.sampling.evaluators.ProductEvaluator
    import scalismo.sampling.proposals.MixtureProposal.implicits._
    import scalismo.sampling.proposals.{MetropolisFilterProposal, MixtureProposal}
    import scalismo.utils.Random

    scalismo.initialize()
    val seed = 1986L
    implicit val rnd: Random = Random(seed)

    val rendererFace12 =  MoMoRenderer(model, RGBA.BlackTransparent).cached(5)

    /* Collection of all pose related proposals */
    def defaultPoseProposal(lmRenderer: ParametricLandmarksRenderer)(implicit rnd: Random):
    ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
      import MixtureProposal.implicits._

      val yawProposalC = GaussianRotationProposal(Vector3D.unitY, 0.75f)
      val yawProposalI = GaussianRotationProposal(Vector3D.unitY, 0.10f)
      val yawProposalF = GaussianRotationProposal(Vector3D.unitY, 0.01f)
      val rotationYaw = MixtureProposal(0.1 *: yawProposalC + 0.4 *: yawProposalI + 0.5 *: yawProposalF)

      val pitchProposalC = GaussianRotationProposal(Vector3D.unitX, 0.75f)
      val pitchProposalI = GaussianRotationProposal(Vector3D.unitX, 0.10f)
      val pitchProposalF = GaussianRotationProposal(Vector3D.unitX, 0.01f)
      val rotationPitch = MixtureProposal(0.1 *: pitchProposalC + 0.4 *: pitchProposalI + 0.5 *: pitchProposalF)

      val rollProposalC = GaussianRotationProposal(Vector3D.unitZ, 0.75f)
      val rollProposalI = GaussianRotationProposal(Vector3D.unitZ, 0.10f)
      val rollProposalF = GaussianRotationProposal(Vector3D.unitZ, 0.01f)
      val rotationRoll = MixtureProposal(0.1 *: rollProposalC + 0.4 *: rollProposalI + 0.5 *: rollProposalF)

      val rotationProposal = MixtureProposal(0.5 *: rotationYaw + 0.3 *: rotationPitch + 0.2 *: rotationRoll).toParameterProposal

      val translationC = GaussianTranslationProposal(Vector(300f, 300f)).toParameterProposal
      val translationF = GaussianTranslationProposal(Vector(50f, 50f)).toParameterProposal
      val translationHF = GaussianTranslationProposal(Vector(10f, 10f)).toParameterProposal
      val translationProposal = MixtureProposal(0.2 *: translationC + 0.2 *: translationF + 0.6 *: translationHF)

      val distanceProposalC = GaussianDistanceProposal(500f, compensateScaling = true).toParameterProposal
      val distanceProposalF = GaussianDistanceProposal(50f, compensateScaling = true).toParameterProposal
      val distanceProposalHF = GaussianDistanceProposal(5f, compensateScaling = true).toParameterProposal
      val distanceProposal = MixtureProposal(0.2 *: distanceProposalC + 0.6 *: distanceProposalF + 0.2 *: distanceProposalHF)

      val scalingProposalC = GaussianScalingProposal(0.15f).toParameterProposal
      val scalingProposalF = GaussianScalingProposal(0.05f).toParameterProposal
      val scalingProposalHF = GaussianScalingProposal(0.01f).toParameterProposal
      val scalingProposal = MixtureProposal(0.2 *: scalingProposalC + 0.6 *: scalingProposalF + 0.2 *: scalingProposalHF)

      val poseMovingNoTransProposal = MixtureProposal(rotationProposal + distanceProposal + scalingProposal)
      val centerREyeProposal = poseMovingNoTransProposal.centeredAt("right.eye.corner_outer", lmRenderer).get
      val centerLEyeProposal = poseMovingNoTransProposal.centeredAt("left.eye.corner_outer", lmRenderer).get
      val centerRLipsProposal = poseMovingNoTransProposal.centeredAt("right.lips.corner", lmRenderer).get
      val centerLLipsProposal = poseMovingNoTransProposal.centeredAt("left.lips.corner", lmRenderer).get

      MixtureProposal(centerREyeProposal + centerLEyeProposal + centerRLipsProposal + centerLLipsProposal + 0.2 *: translationProposal)
    }


    /* Collection of all statistical model (shape, texture) related proposals */
    def neutralMorphableModelProposal(implicit rnd: Random):
    ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {

      val shapeC = GaussianMoMoShapeProposal(0.2f)
      val shapeF = GaussianMoMoShapeProposal(0.1f)
      val shapeHF = GaussianMoMoShapeProposal(0.025f)
      val shapeScaleProposal = GaussianMoMoShapeCaricatureProposal(0.2f)
      val shapeProposal = MixtureProposal(0.1f *: shapeC + 0.5f *: shapeF + 0.2f *: shapeHF + 0.2f *: shapeScaleProposal).toParameterProposal

      val textureC = GaussianMoMoColorProposal(0.2f)
      val textureF = GaussianMoMoColorProposal(0.1f)
      val textureHF = GaussianMoMoColorProposal(0.025f)
      val textureScale = GaussianMoMoColorCaricatureProposal(0.2f)
      val textureProposal = MixtureProposal(0.1f *: textureC + 0.5f *: textureF + 0.2 *: textureHF + 0.2f *: textureScale).toParameterProposal

      MixtureProposal(shapeProposal + textureProposal )
    }

    /* Collection of all statistical model (shape, texture, expression) related proposals */
    def defaultMorphableModelProposal(implicit rnd: Random):
    ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {


      val expressionC = GaussianMoMoExpressionProposal(0.2f)
      val expressionF = GaussianMoMoExpressionProposal(0.1f)
      val expressionHF = GaussianMoMoExpressionProposal(0.025f)
      val expressionScaleProposal = GaussianMoMoExpressionCaricatureProposal(0.2f)
      val expressionProposal = MixtureProposal(0.1f *: expressionC + 0.5f *: expressionF + 0.2f *: expressionHF + 0.2f *: expressionScaleProposal).toParameterProposal


      MixtureProposal(neutralMorphableModelProposal + expressionProposal)
    }

    /* Collection of all color transform proposals */
    def defaultColorProposal(implicit rnd: Random):
    ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
      val colorC = GaussianColorProposal(RGB(0.01f, 0.01f, 0.01f), 0.01f, RGB(1e-4f, 1e-4f, 1e-4f))
      val colorF = GaussianColorProposal(RGB(0.001f, 0.001f, 0.001f), 0.01f, RGB(1e-4f, 1e-4f, 1e-4f))
      val colorHF = GaussianColorProposal(RGB(0.0005f, 0.0005f, 0.0005f), 0.01f, RGB(1e-4f, 1e-4f, 1e-4f))

      MixtureProposal(0.2f *: colorC + 0.6f *: colorF + 0.2f *: colorHF).toParameterProposal
    }

    /* Collection of all illumination related proposals */
    def illuminationProposal(modelRenderer: ParametricImageRenderer[RGBA] with ParametricModel, target: PixelImage[RGBA])(implicit rnd: Random):
    ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {

      val lightSHPert = SHLightPerturbationProposal(0.001f, fixIntensity = true)
      val lightSHIntensity = SHLightIntensityProposal(0.1f)

      val lightSHBandMixter = SHLightBandEnergyMixer(0.1f)
      val lightSHSpatial = SHLightSpatialPerturbation(0.05f)
      val lightSHColor = SHLightColorProposal(0.01f)

      MixtureProposal(lightSHSpatial + lightSHBandMixter + lightSHIntensity + lightSHPert + lightSHColor).toParameterProposal
    }

    // pose proposal
    val totalPose = defaultPoseProposal(rendererFace12)

    //light proposals
    val lightProposal = illuminationProposal(rendererFace12, targetImg)

    //color proposals
    val colorProposal = defaultColorProposal

    //Morphable Model  proposals
    val expression = true
    val momoProposal = if(expression) defaultMorphableModelProposal else neutralMorphableModelProposal

    // Landmarks Evaluator
    val pointEval = IsotropicGaussianPointEvaluator[_2D](4.0) //lm click uncertainty in pixel! -> should be related to image/face size
    val landmarksEval = LandmarkPointEvaluator(targetLMList, pointEval, rendererFace12)

    // Prior Evaluator
    val priorEval = ProductEvaluator(GaussianShapePrior(0, 1), GaussianTexturePrior(0, 1))


    // full proposal filtered by the landmark and prior Evaluator
    val proposal = MetropolisFilterProposal(MetropolisFilterProposal(MixtureProposal(totalPose + colorProposal + 3f*:momoProposal+ 2f *: lightProposal), landmarksEval), priorEval)

    val sdev = 0.043f
    val faceEval = IsotropicGaussianPixelEvaluator(sdev)
    val nonFaceEval = HistogramRGB.fromImageRGBA(targetImg, 25)
    val imgEval = LabeledIndependentPixelEvaluator(targetImg, faceEval, nonFaceEval)
    val labeledModelEval = LabeledImageRendererEvaluator(rendererFace12, imgEval)

    // a dummy segmentation proposal
    class SegmentationProposal(implicit rnd: Random) extends ProposalGenerator[(RenderParameter, PixelImage[Int])]  with SymmetricTransitionRatio[(RenderParameter, PixelImage[Int])] {
      override def propose(current: (RenderParameter, PixelImage[Int])): (RenderParameter, PixelImage[Int]) = current
    }


    // a joint proposal for $\theta$ and $z$ (in this implementation the segmentation proposal is never chosen)
    val masterProposal = SegmentationMasterProposal(proposal, new SegmentationProposal, 1)

    val imageFitter = MetropolisHastings_arneli(masterProposal, labeledModelEval)
    val poseFitter = MetropolisHastings(totalPose, landmarksEval)

    //landmark chain for initialisation
    val initDefault: RenderParameter = RenderParameter.defaultSquare.fitToImageSize(targetImg.width, targetImg.height)
    val init50 = initDefault.withMoMo(initDefault.momo.withNumberOfCoefficients(50, 50, 5))
    val initLMSamples: IndexedSeq[RenderParameter] = poseFitter.iterator(init50).take(5000).toIndexedSeq
    val lmScores = initLMSamples.map(rps => (landmarksEval.logValue(rps), rps))

    val bestLM = lmScores.maxBy(_._1)._2

    println("Landmarks sampled")

    val shOpt = SphericalHarmonicsOptimizer(rendererFace12, targetImg)
    val robustShOptimizerProposal = RobustSHLightSolverProposalWithLabel(rendererFace12, shOpt, targetImg, iterations = 100)
    val dummyImg = targetImg.map(_ => 0)
    val robust = robustShOptimizerProposal.propose(bestLM, dummyImg)

    val labeledPrintLogger = arneli_PrintLogger[(RenderParameter, PixelImage[Int])](Console.out, "").verbose


    //val first1000 = imageFitter.iterator(robust, labeledPrintLogger).take(1000).toIndexedSeq.last
    //println("initial 1000 samples")

    def segmentLBP(target: PixelImage[RGBA],
                   current: (RenderParameter, PixelImage[Int]),
                   renderer: MoMoRenderer): (PixelImage[LabelDistribution]) = {


      val curSample: PixelImage[RGBA] = renderer.renderImage(current._1)

      val nonfaceHist = HistogramRGB.fromImageRGBA(target, 25, 0)
      val nonfaceProb: PixelImage[Double] = target.map(p => nonfaceHist.logValue(p.toRGB))

      val maskedTarget = PixelImage(curSample.domain, (x, y) => RGBA(target(x, y).toRGB, curSample(x, y).a * current._2(x, y).toFloat))
      val fgHist = HistogramRGB.fromImageRGBA(maskedTarget, 25, 0) // replace by hist defined on foreground

      val sdev = 0.043f
      val pixEvalHSV: IsotropicGaussianPixelEvaluatorHSV = IsotropicGaussianPixelEvaluatorHSV(sdev)
      val neighboorhood = 4
      var x: Int = 0
      val fgProbBuffer = PixelImage(nonfaceProb.domain, (_, _) => 0.0).toBuffer

      val curSampleR = curSample.withAccessMode(AccessMode.Repeat())
      while (x < target.width) {
        var y: Int = 0
        while (y < target.height) {
          if (curSample(x, y).a > 0) {
            var maxNeigboorhood = Double.NegativeInfinity
            var q: Int = -neighboorhood
            while (q <= neighboorhood) {
              // Pixels in Face Region F
              var p: Int = -neighboorhood
              while (p <= neighboorhood) {
                val t1 = pixEvalHSV.logValue(target(x, y).toRGB, curSampleR(x + p, y + q).toRGB)
                maxNeigboorhood = Math.max(t1, maxNeigboorhood)
                p += 1
              }
              q += 1
            }
            fgProbBuffer(x, y) = maxNeigboorhood
          }
          else {
            // Pixels in Nonface Region B
            fgProbBuffer(x, y) = fgHist.logValue(target(x, y).toRGB)
          }
          y += 1
        }
        x += 1
      }
      val fgProb: PixelImage[Double] = fgProbBuffer.toImage

      val imageGivenFace = fgProb.map(p => math.exp(p))
      val imageGivenNonFace = nonfaceProb.map(p => math.exp(p))

      val numLabels = 2
      def binDist(pEqual: Double, numLabels: Int, width: Int, height: Int): BinaryLabelDistribution = {
        val pElse = (1 - pEqual) / (numLabels - 1)
        def binDist(k: Label, l: Label) = if (k == l) pEqual else pElse
        PixelImage.view(width, height, (_, _) => binDist)
      }
      val smoothnessDistribution = binDist(0.9, numLabels, target.width, target.height)

      val init = PixelImage(target.width, target.height, (_, _) => Label(0))

      LoopyBPSegmentation.segmentImageFromProb(target.map(_.toRGB), init, imageGivenNonFace, imageGivenFace, smoothnessDistribution, numLabels, 5, gui = false)
    }

    def loadImage(str: String): PixelImage[Int] = {
      val image = PixelImageIO.read[RGBA](new File("data_in/" + str)).get
      val return_mask = PixelImage(image.width, image.height, (x,y) => {
        if(image(x,y).r > 0.5)
          1
        else
          0
      })
      return_mask
    }

    def saveSegmentation(img:PixelImage[Int],filename:String):Unit = {
      val save_file = new File(s"segmentations/" + filename)
      PixelImageIO.write(img.map(RGB(_)), save_file)
    }

    def getDummyMask(target:PixelImage[RGBA]):PixelImage[Int] = {
      val return_mask = PixelImage(target.width, target.height, (x,y) => {
          1 // white
      })
      return_mask
    }

    var current:(RenderParameter,PixelImage[Int]) = null
    if(method == "FCN") {
      current = (initLMSamples(initLMSamples.length - 1), loadImage(filename.replace(".png", "_FCN.png")))
    }else if (method == "GROTRU"){
      current = (initLMSamples(initLMSamples.length - 1), loadImage(filename.replace(".png", "_GROTRU.png")))
    }else if (method == "EGGER"){
      current = imageFitter.iterator(robust, labeledPrintLogger).take(1000).toIndexedSeq.last // In the original version, this was named "first1000"
    }else if (method == "DUMMY"){
      current = (initLMSamples(initLMSamples.length - 1), getDummyMask(targetImg))
    }

    var i = 0
    var numberOfSamples = fitSteps

    val momo = current._1.momo
    val momoNew = momo.withNumberOfCoefficients(model.neutralModel.shape.rank,model.neutralModel.color.rank,model.expressionModel.get.expression.rank)
    current = (current._1.withMoMo(momoNew), current._2)

    var initial_posterior: Double = 0
    while (i < 20) {
      println(s"iteration ${i+1}")

      if(method == "EGGER") {
        // segment
        val zLabel = segmentLBP(targetImg, current, rendererFace12)
        println(s"${i + 1} segmented")
        // update state
        current = (current._1, zLabel.map(_.maxLabel))
      }
      // fit and update state
      current = imageFitter.iterator(current, labeledPrintLogger).take(numberOfSamples).toIndexedSeq.last
      // the above Line takes 99% of the execution time !!!

      saveSegmentation(current._2,s"test${b}_${method}_${i}.png")


      if (i == 0){
        initial_posterior = imageFitter.best_posterior
      }
      bw_pl.write(s"test${b}_${method}_iteration${i}: ${imageFitter.best_posterior-initial_posterior}\n")


      println(s"${i+1} fitted")

      /** Overlay-renderer */
      val renderingFit = MoMoRenderer(model).renderImage(current._1)
      //val fitOverlay = PixelImageOperations.alphaBlending(targetImg.map{_.toRGB}, renderingFit)
      //PixelImageIO.write(fitOverlay, new File(overlayFile.get))


      val fit_file = new File(s"fits/test${b}_${method}_${i}.png")
      PixelImageIO.write(renderingFit, fit_file).get


      val rps_file = new File(s"rps/test${b}_${method}_" + i + ".rps")
      RenderParameterIO.write(current._1, rps_file).get

      i += 1
    }
    current
  }
}

object MetropolisHastings_arneli {
  def apply[A](generator: ProposalGenerator[A] with TransitionRatio[A],
               evaluator: DistributionEvaluator[A])(implicit random: Random) = new arneli_MetropolisHastings[A](generator, evaluator)
}

class arneli_MetropolisHastings[A] (val generator: ProposalGenerator[A] with TransitionRatio[A],
                                    val evaluator: DistributionEvaluator[A])(implicit val random: Random) extends MarkovChain[A] {
  var best_posterior: Double = 0

  private lazy val silentLogger = new SilentLogger[A]()

  /** start a logged iterator */
  def iterator(start: A, logger: AcceptRejectLogger[A]): Iterator[A] = Iterator.iterate(start) { next(_, logger) }

  // next sample
  def next(current: A, logger: AcceptRejectLogger[A]): A = {
    // reference p value
    val currentP = evaluator.logValue(current)
    // proposal
    val proposal = generator.propose(current)
    val proposalP = evaluator.logValue(proposal)
    // transition ratio
    val t = generator.logTransitionRatio(current, proposal)
    // acceptance probability
    val a = proposalP - currentP - t
    // the value to print in the "posterior.txt" file
    best_posterior = if (proposalP > currentP) proposalP else currentP
    // accept or reject
    if (a > 0.0 || random.scalaRandom.nextDouble() < exp(a)) {
      logger.accept(current, proposal, generator, evaluator)
      proposal
    } else {
      logger.reject(current, proposal, generator, evaluator)
      current
    }
  }

  /** next sample in chain */
  override def next(current: A): A = next(current, silentLogger)
}

class arneli_PrintLogger[A](output: PrintStream, prefix: String) extends AcceptRejectLogger[A] {
  override def accept(current: A, sample: A, generator: ProposalGenerator[A], evaluator: DistributionEvaluator[A]): Unit = {
    // Do nothing
  }
  override def reject(current: A, sample: A, generator: ProposalGenerator[A], evaluator: DistributionEvaluator[A]): Unit = {
    // Do nothing
  }

  def verbose = new arneli_VerbosePrintLogger[A](output, prefix)
}

object arneli_PrintLogger{
  def apply[A](output: PrintStream = Console.out, prefix: String = "") = new arneli_PrintLogger[A](output, prefix)
}

class arneli_VerbosePrintLogger[A](output: PrintStream, prefix: String) extends AcceptRejectLogger[A] {
  override def accept(current: A, sample: A, generator: ProposalGenerator[A], evaluator: DistributionEvaluator[A]): Unit = {
    // Do nothing
  }
  override def reject(current: A, sample: A, generator: ProposalGenerator[A], evaluator: DistributionEvaluator[A]): Unit = {
    // Do nothing
  }
}

