package at.ac.oeaw.imba.gerlich.looptrace
package csv

import at.ac.oeaw.imba.gerlich.gerlib.io.csv.ColumnName

/** Collection of names of critical columns from which to parse data */
object ColumnNames:
    private type RoiBag = Set[RoiIndex]

    val RoiIndexColumnName: ColumnName[RoiIndex] = ColumnName[RoiIndex]("index")

    val MergeRoisColumnName: ColumnName[RoiBag] = ColumnName[RoiBag]("mergeRois")
    
    val TooCloseRoisColumnName: ColumnName[RoiBag] = ColumnName[RoiBag]("tooCloseRois")
end ColumnNames