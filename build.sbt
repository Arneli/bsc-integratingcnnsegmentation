name := """bsc-integratingCNNSegmentation"""

version       := "1.0"

scalaVersion  := "2.12.4"

scalacOptions := Seq("-unchecked", "-deprecation", "-encoding", "utf8")

resolvers += Resolver.jcenterRepo

resolvers += Resolver.bintrayRepo("unibas-gravis", "maven")

libraryDependencies += "ch.unibas.cs.gravis" %% "scalismo-faces" % "0.9.0"
libraryDependencies += "ch.unibas.cs.gravis" %% "landmarks-clicker" % "0.2.0"
libraryDependencies += "org.rogach" %% "scallop" % "2.1.3"


mainClass in assembly := Some("FitWithOcclusions")

assemblyJarName in assembly := "bsc-integratingCNNSegmentation.jar"

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}
