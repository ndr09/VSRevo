package it.units.erallab;
/*
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.core.Individual;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.dyn4j.dynamics.Settings;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.Nd4jCpu;

import java.io.*;
import java.util.List;
import java.util.function.Function;

public class NetParser {

    public static void main(String[] args) throws IOException {
        Reader reader = null;
        Writer writer = null;
        List<CSVRecord> records = null;
        List<String> headers = null;
        try {

            reader = new FileReader("me.txt");
            writer = new FileWriter("elaborated.txt");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        CSVParser csvParser = CSVFormat.DEFAULT.withDelimiter(';').withFirstRecordAsHeader().parse(reader);
        records = csvParser.getRecords();
        headers = csvParser.getHeaderNames();
        reader.close();

        Locomotion locomotion = new Locomotion(
                60,
                Locomotion.createTerrain("hilly-3-30-0"),
                new Settings()
        );

        Function<Outcome, Double> getFitness = i -> i.getVelocity();
        Function<Outcome, double[][]> getData = i -> {
            double[][] data = new double[200][];

            double[][] allData = i.getDataObservation();

            System.arraycopy(allData, 0, data, 0, 200);
            return data;
        };
        int c =0;
        for (CSVRecord data: records){
            System.out.println(c);
            c++;
            Integer it = Integer.parseInt(data.get("event→iterations"));
            Robot robot = SerializationUtils.deserialize(data.get("individual→solution→serialized"), Robot.class,SerializationUtils.Mode.GZIPPED_JSON);
            /*if (it < 30){
                MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("0_encoder_2");

                model.setInputMiniBatchSize(1);

                Outcome out = locomotion.apply(robot);
                double[][][][] id = new double[1][1][][];
                id[0][0] = getData.apply(out);
                INDArray idata = Nd4j.create(id);
                model.setInput(idata);
                model.getInputMiniBatchSize();
                double[] desc = model.activateSelectedLayers(0,4,idata).toDoubleVector();
                String o = "2;";
                for (double x: desc){
                    o+= x+";";
                }
                o+= out.getVelocity()+"\n";
                writer.write(o);

            }
            Outcome out = null;
            MultiLayerNetwork model1 = MultiLayerNetwork.load(new File("init"), false);
            MultiLayerNetwork model2 = MultiLayerNetwork.load(new File("0_encoder_9"), false);
            System.out.println(model1.getParam("2_W").eq(model2.getParam("2_W")));
            System.out.println();
            if(it == 49 ||it == 48 ||it == 47) {
                out = locomotion.apply(robot);
            }
            if (it == 49 ||it == 48 ||it == 47){
                MultiLayerNetwork model = MultiLayerNetwork.load(new File("0_encoder_2"), false);
                model.setInputMiniBatchSize(1);
                double[][][][] id = new double[1][1][][];
                id[0][0] = getData.apply(out);
                INDArray idata = Nd4j.create(id);
                model.setInput(idata);
                double[] desc = model.activateSelectedLayers(0,4,idata).toDoubleVector();
                String o = "2;";
                for (double x: desc){
                    o+= x+";";
                }
                o+= out.getVelocity()+"\n";
                writer.write(o);

            }
            if (it == 49 ||it == 48 ||it == 47){
                MultiLayerNetwork model = MultiLayerNetwork.load(new File("0_encoder_3"), false);
                model.setInputMiniBatchSize(1);
                double[][][][] id = new double[1][1][][];
                id[0][0] = getData.apply(out);
                INDArray idata = Nd4j.create(id);
                model.setInput(idata);
                double[] desc = model.activateSelectedLayers(0,4,idata).toDoubleVector();
                String o = "3;";
                for (double x: desc){
                    o+= x+";";
                }
                o+= out.getVelocity()+"\n";
                writer.write(o);

            }
            if (it == 49 ||it == 48 ||it == 47){
                MultiLayerNetwork model = MultiLayerNetwork.load(new File("0_encoder_4"), false);
                model.setInputMiniBatchSize(1);
                double[][][][] id = new double[1][1][][];
                id[0][0] = getData.apply(out);
                INDArray idata = Nd4j.create(id);
                model.setInput(idata);
                double[] desc = model.activateSelectedLayers(0,4,idata).toDoubleVector();
                String o = "4;";
                for (double x: desc){
                    o+= x+";";
                }
                o+= out.getVelocity()+"\n";
                writer.write(o);

            }
            if (it == 49 ||it == 48 ||it == 47){
                MultiLayerNetwork model = MultiLayerNetwork.load(new File("0_encoder_5"), false);
                model.setInputMiniBatchSize(1);
                double[][][][] id = new double[1][1][][];
                id[0][0] = getData.apply(out);
                INDArray idata = Nd4j.create(id);
                model.setInput(idata);
                double[] desc = model.activateSelectedLayers(0,4,idata).toDoubleVector();
                String o = "5;";
                for (double x: desc){
                    o+= x+";";
                }
                o+= out.getVelocity()+"\n";
                writer.write(o);

            }
            if (it == 49 ||it == 48 ||it == 47){
                MultiLayerNetwork model = MultiLayerNetwork.load(new File("0_encoder_6"), false);
                model.setInputMiniBatchSize(1);
                double[][][][] id = new double[1][1][][];
                id[0][0] = getData.apply(out);
                INDArray idata = Nd4j.create(id);
                model.setInput(idata);
                double[] desc = model.activateSelectedLayers(0,4,idata).toDoubleVector();
                String o = "6;";
                for (double x: desc){
                    o+= x+";";
                }
                o+= out.getVelocity()+"\n";
                writer.write(o);

            }
            if (it == 49 ||it == 48 ||it == 47){
                MultiLayerNetwork model = MultiLayerNetwork.load(new File("0_encoder_7"), false);
                model.setInputMiniBatchSize(1);
                double[][][][] id = new double[1][1][][];
                id[0][0] = getData.apply(out);
                INDArray idata = Nd4j.create(id);
                model.setInput(idata);
                double[] desc = model.activateSelectedLayers(0,4,idata).toDoubleVector();
                String o = "7;";
                for (double x: desc){
                    o+= x+";";
                }
                o+= out.getVelocity()+"\n";
                writer.write(o);

            }
            if (it == 49 ||it == 48 ||it == 47){
                MultiLayerNetwork model = MultiLayerNetwork.load(new File("0_encoder_8"), false);
                model.setInputMiniBatchSize(1);
                double[][][][] id = new double[1][1][][];
                id[0][0] = getData.apply(out);
                INDArray idata = Nd4j.create(id);
                model.setInput(idata);
                double[] desc = model.activateSelectedLayers(0,4,idata).toDoubleVector();
                String o = "8;";
                for (double x: desc){
                    o+= x+";";
                }
                o+= out.getVelocity()+"\n";
                writer.write(o);

            }
            if (it == 49 ||it == 48 ||it == 47){
                MultiLayerNetwork model = MultiLayerNetwork.load(new File("0_encoder_9"), false);
                model.setInputMiniBatchSize(1);

                double[][][][] id = new double[1][1][][];
                id[0][0] = getData.apply(out);
                INDArray idata = Nd4j.create(id);
                model.setInput(idata);
                double[] desc = model.activateSelectedLayers(0,4,idata).toDoubleVector();
                String o = "9;";
                for (double x: desc){
                    o+= x+";";
                }
                o+= out.getVelocity()+"\n";
                writer.write(o);

            }
        }
        writer.close();
        reader.close();

    }
}
*/