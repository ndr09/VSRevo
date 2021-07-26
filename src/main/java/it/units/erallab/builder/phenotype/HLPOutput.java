package it.units.erallab.builder.phenotype;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.HebbianPerceptronFullModel;
import it.units.erallab.hmsrobots.core.controllers.HebbianPerceptronOutputModel;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;

import java.util.*;
import java.util.function.Function;

public class HLPOutput implements PrototypedFunctionBuilder<List<Double>, TimedRealFunction> {

    private final HebbianPerceptronOutputModel.ActivationFunction activationFunction;
    private final Random rnd;
    private final double eta;

    public HLPOutput() {
        this(0.1, null, HebbianPerceptronOutputModel.ActivationFunction.TANH);
    }

    public HLPOutput(double eta, Random rnd, HebbianPerceptronOutputModel.ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        this.rnd = rnd;
        this.eta = eta;
    }


    @Override
    public Function<List<Double>, TimedRealFunction> buildFor(TimedRealFunction function) {
        return values -> {
            int nOfInputs = function.getInputDimension();
            int nOfOutputs = function.getOutputDimension();
            int nOfWeights = HebbianPerceptronOutputModel.countHebbCoef(nOfInputs, new int[]{}, nOfOutputs);
            if (nOfWeights != values.size()) {
                throw new IllegalArgumentException(String.format(
                        "Wrong number of values for weights: %d expected, %d found",
                        nOfWeights,
                        values.size()
                ));
            }
            return new HebbianPerceptronOutputModel(
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
                HebbianPerceptronOutputModel.countHebbCoef(
                        function.getInputDimension(),
                        new int[]{},
                        function.getOutputDimension()
                ),
                0d
        );
    }

}
