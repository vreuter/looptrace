package at.ac.oeaw.imba.gerlich.looptrace

import cats.*
import cats.data.NonEmptyList
import cats.effect.IO
import cats.syntax.all.*
import fs2.Stream
import fs2.data.csv.{ CsvRowDecoder, CsvRowEncoder }
import mouse.boolean.*
import scopt.OParser
import com.typesafe.scalalogging.StrictLogging

import at.ac.oeaw.imba.gerlich.gerlib.geometry.{ BoundingBox as BBox }
import at.ac.oeaw.imba.gerlich.gerlib.geometry.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.imaging.ImagingChannel
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.{
    getCsvRowEncoderForSingleton,
    readCsvToCaseClasses, 
    writeCaseClassesToCsv,
}
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnNames.SpotChannelColumnName
import at.ac.oeaw.imba.gerlich.gerlib.io.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given

import at.ac.oeaw.imba.gerlich.looptrace.csv.getCsvRowDecoderForImagingChannel
import at.ac.oeaw.imba.gerlich.looptrace.csv.instances.all.given
import at.ac.oeaw.imba.gerlich.looptrace.cli.scoptReaders.given
import at.ac.oeaw.imba.gerlich.looptrace.internal.BuildInfo
import at.ac.oeaw.imba.gerlich.looptrace.syntax.all.*
import at.ac.oeaw.imba.gerlich.looptrace.roi.{
    MergerAssessedRoi, 
    MergedRoiRecord,
}
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeAndSplitRoiTools.{ 
    IndexedDetectedSpot, 
    PostMergeRoi,
    mergeRois,
}
import at.ac.oeaw.imba.gerlich.looptrace.roi.MergeContributorRoi
import at.ac.oeaw.imba.gerlich.looptrace.space.{ BoundingBox, Point3D }

/** Merge sufficiently proximal FISH spot regions of interest (ROIs). */
object MergeRois extends StrictLogging:
    val ProgramName = "MergeRois"

    final case class CliConfig(
        inputFile: os.Path = null, // unconditionally required
        mergeContributorsFile: os.Path = null, // unconditionally required
        mergeResultsFile: os.Path = null, // unconditionally required
        overwrite: Boolean = false,
    )

    val parserBuilder = OParser.builder[CliConfig]

    def main(args: Array[String]): Unit = {
        import parserBuilder.*

        given Eq[os.Path] = Eq.by(_.toString)

        val parser = OParser.sequence(
            programName(ProgramName), 
            head(ProgramName, BuildInfo.version), 
            opt[os.Path]('I', "inputFile")
                .required()
                .action((f, c) => c.copy(inputFile = f))
                .validate(f => os.isFile(f).either(s"Alleged input path isn't an extant file: $f", ()))
                .text("Path to file from which to read data (merge-assessed ROI records)"),
            opt[os.Path]("mergeContributorsFile")
                .required()
                .action((f, c) => c.copy(mergeContributorsFile = f))
                .validate{ f => os.isDir(f.parent)
                    .either(f"Path to folder for merge contributors file isn't an extant folder: ${f.parent}", ())
                }
                .text("Path to the file to write merge contributor ROIs"),
            opt[os.Path]("mergeResultsFile")
                .required()
                .action((f, c) => c.copy(mergeResultsFile = f))
                .validate{ f => os.isDir(f.parent)
                    .either(f"Path to folder for merge results file isn't an extant folder: ${f.parent}", ())
                }
                .text("Path to the file to write merge contributor ROIs"),
            opt[Unit]("overwrite")
                .action((_, c) => c.copy(overwrite = true))
                .text("Allow overwriting output files."),
            checkConfig{ c => 
                val paths = List(c.inputFile, c.mergeContributorsFile, c.mergeResultsFile)
                if paths.length === paths.toSet.size 
                then success
                else failure(s"Non-unique in/out paths for ROI merge: ${paths}")
            },
            checkConfig{ c => 
                if !c.overwrite && os.exists(c.mergeContributorsFile)
                then failure(s"Overwrite isn't authorised but output file exists: ${c.mergeContributorsFile}")
                else success
            },
            checkConfig{ c => 
                if !c.overwrite && os.exists(c.mergeResultsFile)
                then failure(s"Overwrite isn't authorised but output file exists: ${c.mergeResultsFile}")
                else success
            },
        )

        OParser.parse(parser, args, CliConfig()) match {
            case None => 
                // CLI parser gives error message.
                throw new Exception(s"Illegal CLI use of '${ProgramName}' program. Check --help")
            case Some(opts) => 
                import cats.effect.unsafe.implicits.global // needed for cats.effect.IORuntime
                import fs2.data.text.utf8.* // for CharLikeChunks typeclass instances

                given CsvRowEncoder[ImagingChannel, String] = 
                    getCsvRowEncoderForSingleton(SpotChannelColumnName)

                val writeUnusable: List[MergeContributorRoi] => IO[os.Path] = 
                    val outfile = opts.mergeContributorsFile
                    Stream.emits(_)
                        .through(writeCaseClassesToCsv(outfile))
                        .compile
                        .drain
                        .map(Function.const(outfile))
                
                // using CsvRowEncoder[PostMergeRoi, String]
                val writeUsable: (List[IndexedDetectedSpot], List[MergedRoiRecord]) => IO[os.Path] = 
                    (singletons, merged) => 
                        val outfile = opts.mergeResultsFile
                        val rois: List[PostMergeRoi] = singletons ::: merged
                        Stream.emits(rois)
                            .through(writeCaseClassesToCsv(outfile))
                            .compile
                            .drain
                            .map(Function.const(outfile))

                logger.info(s"Will read ROIs from file: ${opts.inputFile}")
                val prog: IO[Unit] = for {
                    rois <- {
                        given CsvRowDecoder[ImagingChannel, String] = getCsvRowDecoderForImagingChannel(SpotChannelColumnName)
                        readCsvToCaseClasses[MergerAssessedRoi](opts.inputFile)
                    }
                    (individuals, contributors, merged) = 
                        mergeRois(unsafeTakeMaxBoxSize)(rois) match {
                            case (Nil, singletons, contribs, mergers) => 
                                // No errors, so just keep the other collections.
                                (singletons, contribs, mergers)
                            case (es@(e1 :: _), _, _, _) => 
                                // Nonempty errors, so throw exception.
                                throw new Exception(
                                    s"${es.length} error records trying to merge ROIs. First one: ${e1}"
                                )
                        }
                    contributorsFile <- writeUnusable(contributors)
                    _ <- IO{ logger.info(s"Wrote merge contributors file: ${contributorsFile}") }
                    resultRoisFile <- writeUsable(individuals, merged)
                    _ <- IO{ logger.info(s"Wrote post-merge file: ${resultRoisFile}") }
                } yield ()
                prog.unsafeRunSync()
                logger.info("Done!")
        }
    }

    def unsafeTakeMaxBoxSize(center: Point3D, boxes: NonEmptyList[BoundingBox]): BoundingBox = 
        boxes.traverse(BBox.Dimensions.tryFromBox).fold(
            errorMessages => throw new Exception(
                s"${errorMessages.length} error(s) creating max bounding box. First one: ${errorMessages.head}"
            ),
            dimensions => 
                given Semigroup[BBox.Dimensions] with
                    override def combine(a: BBox.Dimensions, b: BBox.Dimensions): BBox.Dimensions = 
                        BBox.Dimensions((a.x, b.x).max, (a.y, b.y).max, (a.z, b.z).max)
                val maxDim = dimensions.reduce
                BBox.around(center)(maxDim)
        )

    extension [A](t: (A, A))
        // Take the max element of a pair.
        private def max(using Order[A]): A = if t._1 > t._2 then t._1 else t._2

end MergeRois
