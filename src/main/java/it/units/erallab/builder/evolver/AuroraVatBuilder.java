package it.units.erallab.builder.evolver;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.AuroraVAT;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.vertex.impl.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;
import java.util.function.Function;

public class AuroraVatBuilder implements EvolverBuilder<List<Double>> {
    private final Mutation<List<Double>> mutation;
    private final int population_size;
    private final int batch_size;
    private final int map_size;
    private final int nc_target;
    private final int seed;

    public AuroraVatBuilder(Mutation<List<Double>> mutation, int map_size, int population_size, int batch_size, int nc_target, int seed) {

        this.mutation = mutation;
        this.map_size = map_size;
        this.population_size = population_size;
        this.batch_size = batch_size;
        this.nc_target = nc_target;
        this.seed =seed;
    }

    @Override
    public <S, F> Evolver<List<Double>, S, F> build(PrototypedFunctionBuilder<List<Double>, S> builder, S target, PartialComparator<F> comparator) {
        int length = builder.exampleFor(target).size();

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(seed)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta())
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.feedForward(29))
                .addLayer("inputl", new DenseLayer.Builder().nIn(29).nOut(15)
                        .build(), "input")
                .addLayer("encoder", new DenseLayer.Builder().nIn(15).nOut(4)
                        .build(), "inputl")
                .addLayer("dec1", new DenseLayer.Builder().nIn(4).nOut(15)
                        .build(), "encoder")
                .addLayer("decfinal", new OutputLayer.Builder().nIn(15).nOut(29)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build(), "dec1")
                .setOutputs("decfinal")
                .build();


        Function<Individual<List<Double>, S, F>, Double> getFitness = i -> ((Outcome) i.getFitness()).getVelocity();
        Function<Individual<List<Double>, S, F>, double[]> getData = i -> {
            return((Outcome) i.getFitness()).getDataObservation().get(0); };

        return new AuroraVAT<>(
                builder.buildFor(target), getFitness,
                getData, new FixedLengthListFactory<>(length, new UniformDoubleFactory(-1d, 1d)),
                comparator.comparing(Individual::getFitness),
                mutation, population_size, map_size, 15, 128, batch_size, nc_target, 25, 10, conf, seed);
    }
}
