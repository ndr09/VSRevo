

package it.units.erallab.builder.evolver;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.AuroraVAT;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;


import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

public class AuroraVatBuilder implements EvolverBuilder<List<Double>> {

    public enum CentertCriterion {ROBOT_CENTER, VOXELS_CENTER, KNN}

    private final Mutation<List<Double>> mutation;
    private final int population_size;
    private final int batch_size;
    private final int map_size;
    private final int nc_target;
    private final int seed;
    private final CentertCriterion criterion;

    public AuroraVatBuilder(Mutation<List<Double>> mutation, int map_size, int population_size, int batch_size, int nc_target, int seed, CentertCriterion dataMode) {

        this.mutation = mutation;
        this.map_size = map_size;
        this.population_size = population_size;
        this.batch_size = batch_size;
        this.nc_target = nc_target;
        this.seed = seed;
        this.criterion = dataMode;
    }

    @Override
    public <S, F> Evolver<List<Double>, S, F> build(PrototypedFunctionBuilder<List<Double>, S> builder, S target, PartialComparator<F> comparator) {
        int length = builder.exampleFor(target).size();


        Function<Individual<List<Double>, S, F>, Double> getFitness = i -> ((Outcome) i.getFitness()).getVelocity();
        Function<Individual<List<Double>, S, F>, double[]> getData;

        int fs = 0;

        switch (criterion) {
            case ROBOT_CENTER -> {
                fs = 2*3600;
                getData = i -> {
                    double[][] data = ((Outcome) i.getFitness()).getCenterPosition();
                    double[] dd = new double[data.length * data[0].length];
                    int c = 0;
                    for (double[] d : ((Outcome) i.getFitness()).getCenterPosition()) {
                        System.arraycopy(d, 0, dd, c * d.length, d.length);
                        c++;
                    }
                    return dd;
                };
                break;
            }
            case VOXELS_CENTER -> {
                fs = 2*3600*((int)((Robot)target).getVoxels().stream().filter(Objects::nonNull).count());
                getData = i -> {
                    double[][] data = ((Outcome) i.getFitness()).getCenterPositions();
                    double[] dd = new double[data.length * data[0].length];
                    int c = 0;
                    for (double[] d : ((Outcome) i.getFitness()).getCenterPosition()) {
                        System.arraycopy(d, 0, dd, c * d.length, d.length);
                        c++;
                    }
                    return dd;
                };
                break;
            }
            case KNN -> {
                getData = i -> null;
                break;
            }
            default -> getData = i -> null;
        }

        BiFunction<Individual<List<Double>, S, F>, double[], double[]> setDesc = (i, d) ->{
            ((Outcome)i.getFitness()).setDesc(d);
            return d;
        };
        Function<Individual<List<Double>, S, F>, double[]> helps = (i) ->{
            return ((Outcome)i.getFitness()).getDesc();
        };
        return new AuroraVAT<>(
                builder.buildFor(target), getFitness,
                getData, setDesc, new FixedLengthListFactory<>(length, new UniformDoubleFactory(-1d, 1d)),
                comparator.comparing(Individual::getFitness),
                mutation, population_size, map_size, 15, 1, batch_size, nc_target, 25, 1, seed,fs, helps );
    }
}