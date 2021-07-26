package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PruningMultiLayerPerceptron;
import it.units.malelab.jgea.core.util.ArrayTable;
import it.units.malelab.jgea.core.util.ImagePlotters;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.core.util.Table;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author eric on 2021/02/10 for VSREvolution
 */
public class MLPPruningTest {

  public static void main(String[] args) throws IOException {
    //nnIOPlots("/home/eric/experiments/2021-gecco-nat-vsr-pruning");
    errPlots("/home/eric/experiments/2021-gecco-nat-vsr-pruning");
  }

  public static void errPlots(String dir) throws IOException {
    int[] nOfInputs = {10, 25, 50};
    int[] nOfLayers = {0, 1, 2};
    int[] seeds = IntStream.range(0, 10).toArray();
    double dT = 1d / 10d;
    double totalT = 10d;
    double[] rates = IntStream.range(20, 31).mapToDouble(i -> (double) i * 0.025d).toArray();
    List<String> localNames = new ArrayList<>();
    localNames.add("rate");
    for (PruningMultiLayerPerceptron.Context context : PruningMultiLayerPerceptron.Context.values()) {
      for (PruningMultiLayerPerceptron.Criterion criterion : PruningMultiLayerPerceptron.Criterion.values()) {
        localNames.add(context.name().toLowerCase() + "/" + criterion.name().toLowerCase());
      }
    }
    List<String> globalNames = new ArrayList<>();
    globalNames.add("nOfInput");
    globalNames.add("nOfLayer");
    globalNames.addAll(localNames);
    Table<Double> globalTable = new ArrayTable<>(globalNames);
    for (int nOfInput : nOfInputs) {
      for (int nOfLayer : nOfLayers) {
        int[] innerLayers = IntStream.range(0, nOfLayer).map(l -> nOfInput).toArray();
        Table<Double> table = new ArrayTable<>(localNames);
        for (double rate : rates) {
          List<Double> row = new ArrayList<>();
          row.add(rate);
          for (PruningMultiLayerPerceptron.Context context : PruningMultiLayerPerceptron.Context.values()) {
            for (PruningMultiLayerPerceptron.Criterion criterion : PruningMultiLayerPerceptron.Criterion.values()) {
              double err = 0d;
              for (int seed : seeds) {
                Random r = new Random(seed);
                MultiLayerPerceptron nn = new MultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH, nOfInput, innerLayers, 1);
                MultiLayerPerceptron pnn = new PruningMultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH, nOfInput, innerLayers, 1, totalT / 2, context, criterion, rate);
                double[] ws = IntStream.range(0, nn.getParams().length).mapToDouble(i -> r.nextDouble() * 2d - 1d).toArray();
                nn.setParams(ws);
                pnn.setParams(ws);
                for (double t = 0; t < totalT; t = t + dT) {
                  final double finalT = t;
                  double[] inputs = IntStream.range(0, nOfInput).mapToDouble(j -> Math.sin(finalT / (double) (j + 1))).toArray();
                  double localErr = Math.abs(nn.apply(inputs)[0] - pnn.apply(t, inputs)[0]);
                  if (t > totalT / 2) {
                    err = err + localErr;
                  }
                }
              }
              row.add(err / (double) seeds.length / totalT * 2d);
            }
          }
          table.addRow(row);
          List<Double> globalRow = new ArrayList<>();
          globalRow.add((double) nOfInput);
          globalRow.add((double) nOfLayer);
          globalRow.addAll(row);
          globalTable.addRow(globalRow);
        }
        String fileName = String.format(
            dir + File.separator + "err-vs-rate-i%d-l%d.png",
            nOfInput,
            innerLayers.length
        );
        System.out.println(fileName);
        ImageIO.write(
            ImagePlotters.xyLines(800, 600).apply(table),
            "png",
            new File(fileName));
      }
    }
    toCSV(globalTable, new File(dir + File.separator + "err.txt"));
  }

  public static void nnIOPlots(String dir) throws IOException {
    int nOfInput = 100;
    int nOfLayer = 2;
    double dT = 1d / 10d;
    double totalT = 20d;
    double pruningT = 5d;
    int[] innerLayers = IntStream.range(0, nOfLayer).map(l -> nOfInput).toArray();
    double[] rates = {0.25, 0.5, 0.75};
    for (PruningMultiLayerPerceptron.Context context : PruningMultiLayerPerceptron.Context.values()) {
      for (PruningMultiLayerPerceptron.Criterion criterion : PruningMultiLayerPerceptron.Criterion.values()) {
        MultiLayerPerceptron nn = new MultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH, nOfInput, innerLayers, 1);
        List<PruningMultiLayerPerceptron> pnns = Arrays.stream(rates).mapToObj(r -> new PruningMultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH, nOfInput, innerLayers, 1, pruningT, context, criterion, r)).collect(Collectors.toList());
        Random r = new Random(2);
        double[] ws = IntStream.range(0, nn.getParams().length).mapToDouble(i -> r.nextDouble() * 2d - 1d).toArray();
        nn.setParams(ws);
        pnns.forEach(pnn -> pnn.setParams(ws));
        List<String> names = new ArrayList<>();
        names.add("x");
        names.add("y-nn");
        for (double rate : rates) {
          names.add(String.format(
              "y-pnn-%s-%s-%4.2f",
              criterion.toString().toLowerCase(),
              context.toString().toLowerCase(),
              rate
          ));
        }
        Table<Double> table = new ArrayTable<>(names);
        for (double t = 0; t < totalT; t = t + dT) {
          double finalT = t;
          double[] inputs = IntStream.range(0, nOfInput).mapToDouble(i -> Math.sin(finalT / (double) (i + 1))).toArray();
          List<Double> values = new ArrayList<>();
          values.add(t);
          values.add(nn.apply(inputs)[0]);
          pnns.forEach(pnn -> values.add(pnn.apply(finalT, inputs)[0]));
          table.addRow(values);
        }
        toCSV(table, new File(String.format(
            dir + File.separator + "pnns-%s-%s-i%d_l%d.txt",
            criterion.toString().toLowerCase(),
            context.toString().toLowerCase(),
            nOfInput,
            nOfLayer
        )));
        ImageIO.write(
            ImagePlotters.xyLines(800, 600).apply(table),
            "png",
            new File(String.format(
                dir + File.separator + "pnns-%s-%s-i%d_l%d.png",
                criterion.toString().toLowerCase(),
                context.toString().toLowerCase(),
                nOfInput,
                nOfLayer
            )));
      }
    }
  }

  private static void toCSV(Table<?> table, File file) throws IOException {
    CSVPrinter printer = new CSVPrinter(new PrintStream(file), CSVFormat.DEFAULT.withDelimiter(';'));
    printer.printRecord(table.names());
    for (List<? extends Pair<String, ?>> row : table.rows()) {
      printer.printRecord(row.stream().map(Pair::second).collect(Collectors.toList()));
    }
    printer.close(true);
  }

}
