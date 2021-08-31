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
    private final double innerLayerRatio;
    private final int nOfInnerLayers;
    private final double[] norm;

    public HLPFull() {
        this(0.1, null, HebbianPerceptronFullModel.ActivationFunction.TANH,0,0, null);
    }

    public HLPFull(double eta, Random rnd, HebbianPerceptronFullModel.ActivationFunction activationFunction, double innerLayerRatio, int nOfInnerLayers, double[] norm) {
        this.activationFunction = activationFunction;
        this.rnd = rnd;
        this.eta = eta;
        this.innerLayerRatio = innerLayerRatio;
        this.nOfInnerLayers = nOfInnerLayers;
        this.norm = norm;
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

            //System.out.println(function);
            int nOfInputs = function.getInputDimension();
            int nOfOutputs = function.getOutputDimension();
            int[] innerNeurons = innerNeurons(nOfInputs, nOfOutputs);
            int nOfWeights = HebbianPerceptronFullModel.countHebbCoef(nOfInputs, innerNeurons, nOfOutputs);
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
        return Collections.nCopies(
                HebbianPerceptronFullModel.countHebbCoef(
                        function.getInputDimension(),
                        innerNeurons(function.getInputDimension(),function.getOutputDimension()),
                        function.getOutputDimension()),
                0d
        );
    }

}
