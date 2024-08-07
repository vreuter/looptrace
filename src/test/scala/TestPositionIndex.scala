package at.ac.oeaw.imba.gerlich.looptrace

import cats.syntax.eq.*
import org.scalacheck.Gen
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should
import org.scalatest.prop.Configuration.PropertyCheckConfiguration
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

import at.ac.oeaw.imba.gerlich.gerlib.numeric.instances.all.given

/** Tests for position index wrapper type */
class TestPositionIndex extends AnyFunSuite, ScalaCheckPropertyChecks, RefinementWrapperSuite, should.Matchers:
    override implicit val generatorDrivenConfig: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 100)
    
    test("Unsafe wrapper works; position index must be nonnegative.") {
        forAll { (z: Int) => 
            if z < 0 then 
                val error = intercept[IllegalArgumentException]{ PositionIndex.unsafe(z) }
                error.getMessage `contains` ironNonnegativityFailureMessage shouldBe true
            else PositionIndex.unsafe(z).get shouldEqual z
        }
    }

    test("Position indices are equivalent on their wrapped values.") {
        forAll (genEquivalenceInputAndExpectation(PositionIndex.apply)) { case (f1, f2, exp) => f1 === f2 shouldBe exp }
    }

    test("Set respects position index equivalence.") {
        forAll (genValuesAndNumUnique(Gen.choose(0, 100))(PositionIndex.unsafe)) { 
            case (indices, expected) => indices.toSet shouldEqual expected
        }
    }
end TestPositionIndex