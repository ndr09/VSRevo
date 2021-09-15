package it.units.erallab.builder.evolver;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.MapElitesEvolver;
import it.units.malelab.jgea.core.evolver.MapElitesWithEvolvabilityEvolver;
import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class MapElitesWithEvolvabiltyBuilder implements EvolverBuilder<List<Double>>  {
    public enum SpeciationCriterion {GENOTYPE, POSTURE, SPECTRUM, GAIT}
    private final static int SPECTRUM_SIZE = 2;
    private final static double SPECTRUM_MIN_FREQ = 0d;
    private final static double SPECTRUM_MAX_FREQ = 5d;

    //private final Function< Individual<List<Double>, T,G>, List<Double>> descriptor;
    private final Mutation<List<Double>> mutation;
    private final int population_size;
    private final int batch_size;
    private final MapElitesBuilder.SpeciationCriterion criterion;
    private final int map_size;


    public MapElitesWithEvolvabiltyBuilder(MapElitesBuilder.SpeciationCriterion criterion, Mutation<List<Double>> mutation, int map_size, int population_size, int batch_size) {
        this.criterion = criterion;
        this.mutation = mutation;
        this.map_size = map_size;
        this.population_size = population_size;
        this.batch_size = batch_size;
    }

    @Override
    public <S, F> Evolver<List<Double>, S, F> build(PrototypedFunctionBuilder<List<Double>, S> builder, S target, PartialComparator<F> comparator) {
        int length = builder.exampleFor(target).size();
        Function<Individual<List<Double>, S, F>, List<Double>> descriptor = switch (criterion) {
            case GENOTYPE ->  i -> new ArrayList<>();
            case POSTURE -> i -> new ArrayList<>();
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
                    ArrayList<Double> descriptors = new ArrayList<>();

                    for (double d :spectrum){
                        descriptors.add(d);
                    }
                    //String s = "";
                    //spectrum = Arrays.stream(spectrum).map(d -> Precision.round(d,2)).toArray();
                    //for(double d : spectrum){
                    //    s += d+" ";
                    //}
                    return descriptors;
                }
                throw new IllegalStateException(String.format("Cannot obtain double[] from %s: Outcome expected", i.getFitness().getClass().getSimpleName()));
            };
            case GAIT -> i -> {
                if (i.getFitness() instanceof Outcome) {
                    Outcome o = (Outcome) i.getFitness();
                    Outcome.Gait gait = o.getMainGait();
                    if (gait == null) {
                        return Arrays.stream(new double[4]).boxed().collect(Collectors.toList());
                    }else {
                        return Arrays.stream(new double[]{
                                gait.getAvgTouchArea(),
                                gait.getCoverage(),
                                gait.getPurity(),
                                gait.getModeInterval()
                        }).boxed().collect(Collectors.toList());
                    }
                }
                throw new IllegalStateException(String.format("Cannot obtain double[] from %s: Outcome expected", i.getFitness().getClass().getSimpleName()));
            };
        };

        ArrayList<Double> mins = new ArrayList<>();
        ArrayList<Double> maxs = new ArrayList<>();
        ArrayList<Integer> sizes = new ArrayList<>();
        for(int i=0; i<SPECTRUM_SIZE*2;i++){
            sizes.add(map_size);
        }
        maxs.add(1d);
        maxs.add(10d);
        maxs.add(1d);
        maxs.add(10d);

        mins.add(0d);
        mins.add(0d);
        mins.add(0d);
        mins.add(0d);
        BiFunction<Individual<List<Double>, S, F>, Integer, double[]> setEvo = (i, d) ->{
            double[] de = new double[1];
            de[0] = d;
            ((Outcome)i.getFitness()).setDesc(de);
            return de;
        };
        return new MapElitesWithEvolvabilityEvolver<>(
                descriptor,
                maxs, mins, sizes,
                builder.buildFor(target),
                new FixedLengthListFactory<>(length, new UniformDoubleFactory(-1d, 1d)),
                comparator.comparing(Individual::getFitness),
                mutation, population_size, batch_size, i -> ((Outcome)i.getFitness()).getVelocity(), setEvo

        );
    }
}
