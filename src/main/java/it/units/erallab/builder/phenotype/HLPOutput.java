package it.units.erallab.builder.phenotype;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.HebbianPerceptronFullModel;
import it.units.erallab.hmsrobots.core.controllers.HebbianMultilayerPerceptronIncomingModel;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;

import java.util.*;
import java.util.function.Function;

public class HLPOutput implements PrototypedFunctionBuilder<List<Double>, TimedRealFunction> {

    private final HebbianMultilayerPerceptronIncomingModel.ActivationFunction activationFunction;
    private final Random rnd;
    private final double eta;
    private final double innerLayerRatio;
    private final int nOfInnerLayers;
    private final double[] norm;

    public HLPOutput() {
        this(0.1, null, HebbianMultilayerPerceptronIncomingModel.ActivationFunction.TANH, 0, 0, null);
    }

    public HLPOutput(double eta, Random rnd, HebbianMultilayerPerceptronIncomingModel.ActivationFunction activationFunction, double innerLayerRatio, int nOfInnerLayers, double[] norm) {
        this.activationFunction = activationFunction;
        this.rnd = rnd;
        this.eta = eta;
        this.innerLayerRatio = innerLayerRatio;
        this.nOfInnerLayers = nOfInnerLayers;
        this.norm =norm;
    }

    private int[] innerNeurons(int nOfInputs, int nOfOutputs) {
        int[] innerNeurons = new int[nOfInnerLayers];
        int centerSize = (int) Math.max(2, Math.round(nOfInputs * innerLayerRatio));
        if (nOfInnerLayers > 1) {
            for (int i = 0; i < nOfInnerLayers / 2; i++) {
                innerNeurons[i] = nOfInputs + (centerSize - nOfInputs) / (nOfInnerLayers / 2 + 1) * (i + 1);
            }
            for (int i = nOfInnerLayers / 2; i < nOfInnerLayers; i++) {
                innerNeurons[i] = centerSize + (nOfOutputs - centerSize) / (nOfInnerLayers / 2 + 1) * (i - nOfInnerLayers / 2);
            }
        } else if (nOfInnerLayers > 0) {
            innerNeurons[0] = centerSize;
        }
        return innerNeurons;
    }

    @Override
    public Function<List<Double>, TimedRealFunction> buildFor(TimedRealFunction function) {
        return values -> {
            int nOfInputs = function.getInputDimension();
            int nOfOutputs = function.getOutputDimension();
            int[] innerNeurons = innerNeurons(nOfInputs, nOfOutputs);
            int nOfWeights = HebbianMultilayerPerceptronIncomingModel.countHebbCoef(nOfInputs, innerNeurons, nOfOutputs);
            if (nOfWeights != values.size()) {
                throw new IllegalArgumentException(String.format(
                        "Wrong number of values for weights: %d expected, %d found",
                        nOfWeights,
                        values.size()
                ));
            }
            return new HebbianMultilayerPerceptronIncomingModel(
                    activationFunction,
                    nOfInputs,
                    innerNeurons,
                    nOfOutputs,
                    values.stream().mapToDouble(d -> d).toArray(),
                    eta,
                    rnd,
                    new HashSet<Integer>(),
                    new HashMap<Integer, Integer>(),
                    norm);
        };
    }


    @Override
    public List<Double> exampleFor(TimedRealFunction function) {
        int[] innerNeurons = innerNeurons(function.getInputDimension(), function.getOutputDimension());
        return Collections.nCopies(
                HebbianMultilayerPerceptronIncomingModel.countHebbCoef(
                        function.getInputDimension(),
                        innerNeurons,
                        function.getOutputDimension()
                ),
                0d
        );
    }

}
