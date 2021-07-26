package it.units.erallab.builder.evolver;

import com.google.common.collect.Range;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.speciation.KMeansSpeciator;
import it.units.malelab.jgea.core.evolver.speciation.SpeciatedEvolver;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.distance.Distance;
import it.units.malelab.jgea.distance.LNorm;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.GeometricCrossover;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * @author eric
 */
public class DoublesSpeciated implements EvolverBuilder<List<Double>> {

  private final static int SPECTRUM_SIZE = 8;
  private final static double SPECTRUM_MIN_FREQ = 0d;
  private final static double SPECTRUM_MAX_FREQ = 5d;

  public enum SpeciationCriterion {GENOTYPE, POSTURE, SPECTRUM, GAIT}

  private final int nPop;
  private final int nSpecies;
  private final double xOverProb;
  private final SpeciationCriterion criterion;

  public DoublesSpeciated(int nPop, int nSpecies, double xOverProb, SpeciationCriterion criterion) {
    this.nPop = nPop;
    this.nSpecies = nSpecies;
    this.xOverProb = xOverProb;
    this.criterion = criterion;
  }

  @Override
  public <T, F> Evolver<List<Double>, T, F> build(PrototypedFunctionBuilder<List<Double>, T> builder, T target, PartialComparator<F> comparator) {
    Function<Individual<List<Double>, T, F>, double[]> converter = switch (criterion) {
      case GENOTYPE -> i -> i.getGenotype().stream().mapToDouble(Double::doubleValue).toArray();
      case POSTURE -> i -> {
        if (i.getFitness() instanceof Outcome) {
          Outcome o = (Outcome) i.getFitness();
          return o.getAveragePosture().values().stream().mapToDouble(b -> b ? 1d : 0d).toArray();
        }
        throw new IllegalStateException(String.format("Cannot obtain double[] from %s: Outcome expected", i.getFitness().getClass().getSimpleName()));
      };
      case SPECTRUM -> i -> {
        if (i.getFitness() instanceof Outcome) {
          Outcome o = (Outcome) i.getFitness();
          double[] xSpectrum = o.getCenterPowerSpectrum(Outcome.Component.X, SPECTRUM_MIN_FREQ, SPECTRUM_MAX_FREQ, SPECTRUM_SIZE).stream()
              .mapToDouble(Outcome.Mode::getStrength)
              .toArray();
          double[] ySpectrum = o.getCenterPowerSpectrum(Outcome.Component.Y, SPECTRUM_MIN_FREQ, SPECTRUM_MAX_FREQ, SPECTRUM_SIZE).stream()
              .mapToDouble(Outcome.Mode::getStrength)
              .toArray();
          double[] spectrum = new double[SPECTRUM_SIZE * 2];
          System.arraycopy(xSpectrum, 0, spectrum, 0, SPECTRUM_SIZE);
          System.arraycopy(ySpectrum, 0, spectrum, SPECTRUM_SIZE, SPECTRUM_SIZE);
          return spectrum;
        }
        throw new IllegalStateException(String.format("Cannot obtain double[] from %s: Outcome expected", i.getFitness().getClass().getSimpleName()));
      };
      case GAIT -> i -> {
        if (i.getFitness() instanceof Outcome) {
          Outcome o = (Outcome) i.getFitness();
          Outcome.Gait gait = o.getMainGait();
          if (gait == null) {
            return new double[4];
          }
          return new double[]{
              gait.getAvgTouchArea(),
              gait.getCoverage(),
              gait.getModeInterval(),
              gait.getPurity()
          };
        }
        throw new IllegalStateException(String.format("Cannot obtain double[] from %s: Outcome expected", i.getFitness().getClass().getSimpleName()));
      };
    };
    int length = builder.exampleFor(target).size();
    return new SpeciatedEvolver<>(
        builder.buildFor(target),
        new FixedLengthListFactory<>(length, new UniformDoubleFactory(-1d, 1d)),
        comparator.comparing(Individual::getFitness),
        nPop,
        Map.of(
            new GaussianMutation(.35d), 1d - xOverProb,
            new GeometricCrossover(Range.closed(-.5d, 1.5d)).andThen(new GaussianMutation(.1d)), xOverProb
        ),
        nPop / nSpecies,
        new KMeansSpeciator<>(
            nSpecies,
            -1,
            new LNorm(2),
            converter
        ),
        0.75d,
        true
    );
  }

}
