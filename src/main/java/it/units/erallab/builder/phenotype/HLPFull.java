package it.units.erallab.builder.phenotype;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.HebbianPerceptronFullModel;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;

import java.util.*;
import java.util.function.Function;

public class HLPFull implements PrototypedFunctionBuilder<List<Double>, TimedRealFunction> {

    private final HebbianPerceptronFullModel.ActivationFunction activationFunction;
    private final Random rnd;
    private final double eta;


    public HLPFull() {
        this(0.1, null, HebbianPerceptronFullModel.ActivationFunction.TANH);
    }

    public HLPFull(double eta, Random rnd, HebbianPerceptronFullModel.ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        this.rnd = rnd;
        this.eta = eta;
    }


    @Override
    public Function<List<Double>, TimedRealFunction> buildFor(TimedRealFunction function) {
        return values -> {

            //System.out.println(function);
            int nOfInputs = function.getInputDimension();
            int nOfOutputs = function.getOutputDimension();
            System.out.println("size " + values.size());
            int nOfWeights = HebbianPerceptronFullModel.countHebbCoef(nOfInputs, new int[]{}, nOfOutputs);
            if (nOfWeights != values.size()) {
                throw new IllegalArgumentException(String.format(
                        "Wrong number of values for weights: %d expected, %d found",
                        nOfWeights,
                        values.size()
                ));
            }
            return new HebbianPerceptronFullModel(
                    activationFunction,
                    nOfInputs,
                    new int[]{},
                    nOfOutputs,
                    values.stream().mapToDouble(d -> d).toArray(),
                    eta,
                    rnd,
                    new HashSet<Integer>(),
                    new HashMap<Integer, Integer>());
        };
    }


    @Override
    public List<Double> exampleFor(TimedRealFunction function) {
        return Collections.nCopies(
                HebbianPerceptronFullModel.countHebbCoef(
                        function.getInputDimension(),
                        new int[]{},
                        function.getOutputDimension()),
                0d
        );
    }

}
