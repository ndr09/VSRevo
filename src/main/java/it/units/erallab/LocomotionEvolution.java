/*
 * Copyright 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.units.erallab;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import it.units.erallab.builder.DirectNumbersGrid;
import it.units.erallab.builder.FunctionGrid;
import it.units.erallab.builder.FunctionNumbersGrid;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.evolver.*;
import it.units.erallab.builder.phenotype.*;
import it.units.erallab.builder.robot.*;
import it.units.erallab.hmsrobots.core.controllers.HebbianPerceptronFullModel;
import it.units.erallab.hmsrobots.core.controllers.HebbianPerceptronOutputModel;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PruningMultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.MapElitesEvolver;
import it.units.malelab.jgea.core.evolver.stopcondition.FitnessEvaluations;
import it.units.malelab.jgea.core.listener.*;
import it.units.malelab.jgea.core.listener.telegram.TelegramUpdater;
import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.core.order.MapElites;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.core.util.SequentialFunction;
import it.units.malelab.jgea.core.util.TextPlotter;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;

import static it.units.erallab.hmsrobots.util.Utils.params;
import static it.units.malelab.jgea.core.listener.NamedFunctions.*;
import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author eric
 */
public class LocomotionEvolution extends Worker {

    private final static Settings PHYSICS_SETTINGS = new Settings();

    public static class ValidationOutcome {
        private final Event<?, ? extends Robot<?>, ? extends Outcome> event;
        private final Map<String, Object> keys;
        private final Outcome outcome;

        public ValidationOutcome(Event<?, ? extends Robot<?>, ? extends Outcome> event, Map<String, Object> keys, Outcome outcome) {
            this.event = event;
            this.keys = keys;
            this.outcome = outcome;
        }
    }

    public static final int CACHE_SIZE = 1000;
    public static final String MAPPER_PIPE_CHAR = "<";
    public static final String SEQUENCE_SEPARATOR_CHAR = ">";
    public static final String SEQUENCE_ITERATION_CHAR = ":";

    public LocomotionEvolution(String[] args) {
        super(args);
    }

    public static void main(String[] args) {
        new LocomotionEvolution(args);
    }

    @Override
    public void run() {
        int spectrumSize = 2;
        double spectrumMinFreq = 0d;
        double spectrumMaxFreq = 5d;
        double episodeTime = d(a("episodeTime", "1"));
        double episodeTransientTime = d(a("episodeTransientTime", "0"));
        double validationEpisodeTime = d(a("validationEpisodeTime", Double.toString(episodeTime)));
        double validationEpisodeTransientTime = d(a("validationEpisodeTransientTime", Double.toString(episodeTransientTime)));
        double videoEpisodeTime = d(a("videoEpisodeTime", "10"));
        double videoEpisodeTransientTime = d(a("videoEpisodeTransientTime", "0"));
        int nEvals = i(a("nEvals", "1"));
        int seed = i(a("seed", "0"));
        String experimentName = a("expName", "short");
        List<String> terrainNames = l(a("terrain", "hilly-1-30-0"));//"hilly-1-10-rnd"));
        List<String> targetShapeNames = l(a("shape", "biped-4x3"));
        List<String> targetSensorConfigNames = l(a("sensorConfig", "high_biped-0.01-f"));
        List<String> transformationNames = l(a("transformation", "identity"));
        //"auroraVat-(?<sigma>\\d+(\\.\\d+)?)-(?<ms>\\d+)-(?<nPop>\\d+)-(?<bs>\\d+)-(?<nc_target>\\d+)"; auroraVat-0.1-4-1-1-10-0
        List<String> evolverNames = l(a("evolver", "ES-1-0.1"));
        //HLP-(?<type>(full|output))-(?<eta>\d+(\.\d+)?)(-(?<actFun>(tanh|sigmoid|relu)))?(-(?<seed>\d+)))?
        List<String> mapperNames = l(a("mapper", "fixedCentralized<HLP-full-0.1-tanh-0.1-1"));
        String lastFileName = a("lastFile", "last.txt");
        String bestFileName = a("bestFile", "best.txt");
        String allFileName = a("allFile", null);
        String allMEFileName = a("allMEFile", "testing.txt");
        String validationFileName = a("validationFile", "val.txt");
        boolean deferred = a("deferred", "true").startsWith("t");
        String telegramBotId = a("telegramBotId", null);
        long telegramChatId = Long.parseLong(a("telegramChatId", "0"));
        List<String> serializationFlags = l(a("serialization", "last,best,all")); //last,best,all
        boolean output = a("output", "false").startsWith("t");
        List<String> validationTransformationNames = l(a("validationTransformation", "")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
        List<String> validationTerrainNames = l(a("validationTerrain", "flat,downhill-30")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
        Function<Outcome, Double> fitnessFunction = Outcome::getVelocity;
        //consumers
        List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> keysFunctions = Utils.keysFunctions();
        List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> basicFunctions = Utils.basicFunctions();
        List<NamedFunction<Individual<?, ? extends Robot<?>, ? extends Outcome>, ?>> basicIndividualFunctions = Utils.individualFunctions(fitnessFunction);
        List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> populationFunctions = Utils.populationFunctions(fitnessFunction);
        List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> visualFunctions = Utils.visualFunctions(fitnessFunction);
        List<NamedFunction<Outcome, ?>> basicOutcomeFunctions = Utils.basicOutcomeFunctions();
        List<NamedFunction<Outcome, ?>> detailedOutcomeFunctions = Utils.detailedOutcomeFunctions(spectrumMinFreq, spectrumMaxFreq, spectrumSize);
        List<NamedFunction<Outcome, ?>> mapEliteXYFunctions = Utils.visualOutcomeFunctions(spectrumMinFreq, spectrumMaxFreq);
        List<NamedFunction<Outcome, ?>> visualOutcomeFunctions = Utils.visualOutcomeFunctions(spectrumMinFreq, spectrumMaxFreq);
        Listener.Factory<Event<?, ? extends Robot<?>, ? extends Outcome>> factory = Listener.Factory.deaf();
        //screen listener
        if (bestFileName == null || output) {
            factory = factory.and(new TabularPrinter<>(Misc.concat(List.of(
                    basicFunctions,
                    populationFunctions,
                    visualFunctions,
                    NamedFunction.then(best(), basicIndividualFunctions),
                    NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), basicOutcomeFunctions),
                    NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), visualOutcomeFunctions)
            ))));
        }
        //file listeners
        if (lastFileName != null) {
            factory = factory.and(new CSVPrinter<>(Misc.concat(List.of(
                    keysFunctions,
                    basicFunctions,
                    populationFunctions,
                    NamedFunction.then(best(), basicIndividualFunctions),
                    NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), basicOutcomeFunctions),
                    NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), detailedOutcomeFunctions),
                    NamedFunction.then(best(), Utils.serializationFunction(serializationFlags.contains("last")))
            )), new File(lastFileName)
            ).onLast());
        }
        if (bestFileName != null) {
            factory = factory.and(new CSVPrinter<>(Misc.concat(List.of(
                    keysFunctions,
                    basicFunctions,
                    populationFunctions,
                    NamedFunction.then(best(), basicIndividualFunctions),
                    NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), basicOutcomeFunctions),
                    NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), detailedOutcomeFunctions),
                    NamedFunction.then(best(), Utils.serializationFunction(serializationFlags.contains("best")))
            )), new File(bestFileName)
            ));
        }
        if (allFileName != null) {
            factory = factory.and(Listener.Factory.forEach(
                    event -> event.getOrderedPopulation().all().stream()
                            .map(i -> Pair.of(event, i))
                            .collect(Collectors.toList()),
                    new CSVPrinter<>(
                            Misc.concat(List.of(
                                    NamedFunction.then(f("event", Pair::first), keysFunctions),
                                    NamedFunction.then(f("event", Pair::first), basicFunctions),
                                    NamedFunction.then(f("individual", Pair::second), basicIndividualFunctions)
                                    //NamedFunction.then(f("individual", Pair::second), Utils.serializationFunction(serializationFlags.contains("all")))
                            )),
                            new File(allFileName)
                    )
            ));
        }
        if (allMEFileName != null) {
            factory = factory.and(Listener.Factory.forEach(
                    event -> event.getSerializationPop().all().stream()
                            .map(i -> Pair.of(event, i))
                            .collect(Collectors.toList()),
                    new CSVPrinter<>(
                            Misc.concat(List.of(
                                    NamedFunction.then(f("event", Pair::first), keysFunctions),
                                    NamedFunction.then(f("event", Pair::first), basicFunctions),
                                    NamedFunction.then(f("individual", Pair::second), basicIndividualFunctions),
                                    NamedFunction.then(f("individual", Pair::second), Utils.serializationFunction(serializationFlags.contains("all")))
                            )),
                            new File(allMEFileName)
                    )
            ));
        }
        //validation listener
        if (validationFileName != null) {
            if (!validationTerrainNames.isEmpty() && validationTransformationNames.isEmpty()) {
                validationTransformationNames.add("identity");
            }
            if (validationTerrainNames.isEmpty() && !validationTransformationNames.isEmpty()) {
                validationTerrainNames.add(terrainNames.get(0));
            }
            Listener.Factory<Event<?, ? extends Robot<?>, ? extends Outcome>> validationFactory = Listener.Factory.forEach(
                    Utils.validation(validationTerrainNames, validationTransformationNames, List.of(0), validationEpisodeTime),
                    new CSVPrinter<>(
                            Misc.concat(List.of(
                                    NamedFunction.then(f("event", (ValidationOutcome vo) -> vo.event), basicFunctions),
                                    NamedFunction.then(f("event", (ValidationOutcome vo) -> vo.event), keysFunctions),
                                    NamedFunction.then(f("keys", (ValidationOutcome vo) -> vo.keys), List.of(
                                            f("validation.terrain", (Map<String, Object> map) -> map.get("validation.terrain")),
                                            f("validation.transformation", (Map<String, Object> map) -> map.get("validation.transformation")),
                                            f("validation.seed", "%2d", (Map<String, Object> map) -> map.get("validation.seed"))
                                    )),
                                    NamedFunction.then(
                                            f("outcome", (ValidationOutcome vo) -> vo.outcome.subOutcome(validationEpisodeTransientTime, validationEpisodeTime)),
                                            basicOutcomeFunctions
                                    ),
                                    NamedFunction.then(
                                            f("outcome", (ValidationOutcome vo) -> vo.outcome.subOutcome(validationEpisodeTransientTime, validationEpisodeTime)),
                                            detailedOutcomeFunctions
                                    )
                            )),
                            new File(validationFileName)
                    )
            ).onLast();
            factory = factory.and(validationFactory);
        }
        if (telegramBotId != null && telegramChatId != 0) {
            factory = factory.and(new TelegramUpdater<>(List.of(
                    Utils.lastEventToString(fitnessFunction),
                    Utils.fitnessPlot(fitnessFunction),
                    Utils.centerPositionPlot(),
                    Utils.bestVideo(videoEpisodeTransientTime, videoEpisodeTime, PHYSICS_SETTINGS)
            ), telegramBotId, telegramChatId));
        }
        //summarize params
        L.info("Experiment name: " + experimentName);
        L.info("Evolvers: " + evolverNames);
        L.info("Mappers: " + mapperNames);
        L.info("Shapes: " + targetShapeNames);
        L.info("Sensor configs: " + targetSensorConfigNames);
        L.info("Terrains: " + terrainNames);
        L.info("Transformations: " + transformationNames);
        L.info("Validations: " + Lists.cartesianProduct(validationTerrainNames, validationTransformationNames));
        //start iterations
        int nOfRuns = terrainNames.size() * targetShapeNames.size() * targetSensorConfigNames.size() * mapperNames.size() * transformationNames.size() * evolverNames.size();
        int counter = 0;
        for (String terrainName : terrainNames) {
            for (String targetShapeName : targetShapeNames) {
                for (String targetSensorConfigName : targetSensorConfigNames) {
                    for (String mapperName : mapperNames) {
                        for (String transformationName : transformationNames) {
                            for (String evolverName : evolverNames) {
                                counter = counter + 1;
                                final Random random = new Random(seed);
                                //prepare keys
                                Map<String, Object> keys = Map.ofEntries(
                                        Map.entry("experiment.name", experimentName),
                                        Map.entry("seed", seed),
                                        Map.entry("terrain", terrainName),
                                        Map.entry("shape", targetShapeName),
                                        Map.entry("sensor.config", targetSensorConfigName),
                                        Map.entry("mapper", mapperName),
                                        Map.entry("transformation", transformationName),
                                        Map.entry("evolver", evolverName)
                                );
                                Robot<?> target = new Robot<>(
                                        null,
                                        RobotUtils.buildSensorizingFunction(targetSensorConfigName).apply(RobotUtils.buildShape(targetShapeName))
                                );
                                //build evolver
                                Evolver<?, Robot<?>, Outcome> evolver;
                                try {
                                    evolver = buildEvolver(evolverName, mapperName, target, fitnessFunction);
                                } catch (ClassCastException | IllegalArgumentException e) {
                                    L.warning(String.format(
                                            "Cannot instantiate %s for %s: %s",
                                            evolverName,
                                            mapperName,
                                            e.toString()
                                    ));
                                    continue;
                                }
                                Listener<Event<?, ? extends Robot<?>, ? extends Outcome>> listener = Listener.all(List.of(
                                        new EventAugmenter(keys),
                                        factory.build()
                                ));
                                if (deferred) {
                                    listener = listener.deferred(executorService);
                                }
                                //optimize
                                Stopwatch stopwatch = Stopwatch.createStarted();
                                L.info(String.format("Progress %s (%d/%d); Starting %s",
                                        TextPlotter.horizontalBar(counter - 1, 0, nOfRuns, 8),
                                        counter, nOfRuns,
                                        keys
                                ));
                                //build task
                                try {
                                    Collection<Robot<?>> solutions = evolver.solve(
                                            buildTaskFromName(transformationName, terrainName, episodeTime, random).andThen(o -> o.subOutcome(episodeTransientTime, episodeTime)),
                                            new FitnessEvaluations(nEvals),
                                            random,
                                            executorService,
                                            listener
                                    );
                                    L.info(String.format("Progress %s (%d/%d); Done: %d solutions in %4ds",
                                            TextPlotter.horizontalBar(counter, 0, nOfRuns, 8),
                                            counter, nOfRuns,
                                            solutions.size(),
                                            stopwatch.elapsed(TimeUnit.SECONDS)
                                    ));
                                } catch (Exception e) {
                                    L.severe(String.format("Cannot complete %s due to %s",
                                            keys,
                                            e
                                    ));
                                    e.printStackTrace(); // TODO possibly to be removed
                                }
                            }
                        }
                    }
                }
            }
        }
        factory.shutdown();
    }

    private static EvolverBuilder<?> getEvolverBuilderFromName(String name) {
        String numGA = "numGA-(?<nPop>\\d+)-(?<diversity>(t|f))";
        String numGASpeciated = "numGASpec-(?<nPop>\\d+)-(?<nSpecies>\\d+)-(?<criterion>(" + Arrays.stream(DoublesSpeciated.SpeciationCriterion.values()).map(c -> c.name().toLowerCase(Locale.ROOT)).collect(Collectors.joining("|")) + "))";
        String cmaES = "CMAES";
        String mapElites = "mapElites-(?<sigma>\\d+(\\.\\d+)?)-(?<ms>\\d+)-(?<nPop>\\d+)-(?<bs>\\d+)-(?<criterion>(" + Arrays.stream(DoublesSpeciated.SpeciationCriterion.values()).map(c -> c.name().toLowerCase(Locale.ROOT)).collect(Collectors.joining("|")) + "))";
        String auroraVat = "auroraVat-(?<sigma>\\d+(\\.\\d+)?)-(?<ms>\\d+)-(?<nPop>\\d+)-(?<bs>\\d+)-(?<nctarget>\\d+)-(?<seed>\\d+)";
        String eS = "ES-(?<nPop>\\d+)-(?<sigma>\\d+(\\.\\d+)?)";
        Map<String, String> params;

        if((params = params(auroraVat, name)) != null) {
            return new AuroraVatBuilder(new GaussianMutation(Double.parseDouble(params.get("sigma"))),
                    Integer.parseInt(params.get("ms")),
                    Integer.parseInt(params.get("nPop")),
                    Integer.parseInt(params.get("bs")),
                    Integer.parseInt(params.get("nctarget")),
                    Integer.parseInt(params.get("seed")));
        }

        if ((params = params(mapElites, name)) != null) {
            return new MapElitesBuilder(
                    MapElitesBuilder.SpeciationCriterion.valueOf(params.get("criterion").toUpperCase()),
                    new GaussianMutation(Double.parseDouble(params.get("sigma"))),
                    Integer.parseInt(params.get("ms")),
                    Integer.parseInt(params.get("nPop")),
                    Integer.parseInt(params.get("bs")));
        }

        if ((params = params(numGA, name)) != null) {
            return new DoublesStandard(
                    Integer.parseInt(params.get("nPop")),
                    (int) Math.max(Math.round((double) Integer.parseInt(params.get("nPop")) / 10d), 3),
                    0.75d,
                    params.get("diversity").equals("t")
            );
        }
        if ((params = params(numGASpeciated, name)) != null) {
            return new DoublesSpeciated(
                    Integer.parseInt(params.get("nPop")),
                    Integer.parseInt(params.get("nSpecies")),
                    0.75d,
                    DoublesSpeciated.SpeciationCriterion.valueOf(params.get("criterion").toUpperCase())
            );
        }
        if ((params = params(eS, name)) != null) {
            return new ES(
                    Double.parseDouble(params.get("sigma")),
                    Integer.parseInt(params.get("nPop"))
            );
        }
        if ((params = params(cmaES, name)) != null) {
            return new CMAES();
        }
        throw new IllegalArgumentException(String.format("Unknown evolver builder name: %s", name));
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static PrototypedFunctionBuilder<?, ?> getMapperBuilderFromName(String name) {
        String fixedCentralized = "fixedCentralized";
        String fixedHomoDistributed = "fixedHomoDist-(?<nSignals>\\d+)";
        String fixedHeteroDistributed = "fixedHeteroDist-(?<nSignals>\\d+)";
        String fixedPhasesFunction = "fixedPhasesFunct-(?<f>\\d+)";
        String fixedPhases = "fixedPhases-(?<f>\\d+)";
        String bodySin = "bodySin-(?<fullness>\\d+(\\.\\d+)?)-(?<minF>\\d+(\\.\\d+)?)-(?<maxF>\\d+(\\.\\d+)?)";
        String bodyAndHomoDistributed = "bodyAndHomoDist-(?<fullness>\\d+(\\.\\d+)?)-(?<nSignals>\\d+)-(?<nLayers>\\d+)";
        String sensorAndBodyAndHomoDistributed = "sensorAndBodyAndHomoDist-(?<fullness>\\d+(\\.\\d+)?)-(?<nSignals>\\d+)-(?<nLayers>\\d+)-(?<position>(t|f))";
        String sensorCentralized = "sensorCentralized-(?<nLayers>\\d+)";
        String mlp = "MLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)(-(?<actFun>(sin|tanh|sigmoid|relu)))?";
        String hlp = "HLP-(?<type>(full|output))-(?<eta>\\d+(\\.\\d+)?)(-(?<actFun>(tanh|sigmoid|relu)))-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)?(-(?<seed>\\d+))?";
        String pruningMlp = "pMLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<actFun>(sin|tanh|sigmoid|relu))-(?<pruningTime>\\d+(\\.\\d+)?)-(?<pruningRate>0(\\.\\d+)?)-(?<criterion>(weight|abs_signal_mean|random))";
        String directNumGrid = "directNumGrid";
        String functionNumGrid = "functionNumGrid";
        String fgraph = "fGraph";
        String functionGrid = "fGrid-(?<innerMapper>.*)";
        Map<String, String> params;
        //robot mappers
        if ((params = params(fixedCentralized, name)) != null) {
            return new FixedCentralized();
        }
        if ((params = params(fixedHomoDistributed, name)) != null) {
            return new FixedHomoDistributed(
                    Integer.parseInt(params.get("nSignals"))
            );
        }
        if ((params = params(fixedHeteroDistributed, name)) != null) {
            return new FixedHeteroDistributed(
                    Integer.parseInt(params.get("nSignals"))
            );
        }
        if ((params = params(fixedPhasesFunction, name)) != null) {
            return new FixedPhaseFunction(
                    Double.parseDouble(params.get("f")),
                    1d
            );
        }
        if ((params = params(fixedPhases, name)) != null) {
            return new FixedPhaseValues(
                    Double.parseDouble(params.get("f")),
                    1d
            );
        }
        if ((params = params(bodyAndHomoDistributed, name)) != null) {
            return new BodyAndHomoDistributed(
                    Integer.parseInt(params.get("nSignals")),
                    Double.parseDouble(params.get("fullness"))
            )
                    .compose(PrototypedFunctionBuilder.of(List.of(
                            new MLP(2d, 3, MultiLayerPerceptron.ActivationFunction.SIN),
                            new MLP(0.65d, Integer.parseInt(params.get("nLayers")))
                    )))
                    .compose(PrototypedFunctionBuilder.merger());
        }
        if ((params = params(sensorAndBodyAndHomoDistributed, name)) != null) {
            return new SensorAndBodyAndHomoDistributed(
                    Integer.parseInt(params.get("nSignals")),
                    Double.parseDouble(params.get("fullness")),
                    params.get("position").equals("t")
            )
                    .compose(PrototypedFunctionBuilder.of(List.of(
                            new MLP(2d, 3, MultiLayerPerceptron.ActivationFunction.SIN),
                            new MLP(1.5d, Integer.parseInt(params.get("nLayers")))
                    )))
                    .compose(PrototypedFunctionBuilder.merger());
        }
        if ((params = params(bodySin, name)) != null) {
            return new BodyAndSinusoidal(
                    Double.parseDouble(params.get("minF")),
                    Double.parseDouble(params.get("maxF")),
                    Double.parseDouble(params.get("fullness")),
                    Set.of(BodyAndSinusoidal.Component.FREQUENCY, BodyAndSinusoidal.Component.PHASE, BodyAndSinusoidal.Component.AMPLITUDE)
            );
        }
        if ((params = params(fixedHomoDistributed, name)) != null) {
            return new FixedHomoDistributed(
                    Integer.parseInt(params.get("nSignals"))
            );
        }
        if ((params = params(sensorCentralized, name)) != null) {
            return new SensorCentralized()
                    .compose(PrototypedFunctionBuilder.of(List.of(
                            new MLP(2d, 3, MultiLayerPerceptron.ActivationFunction.SIN),
                            new MLP(1.5d, Integer.parseInt(params.get("nLayers")))
                    )))
                    .compose(PrototypedFunctionBuilder.merger());
        }
        //function mappers
        if ((params = params(mlp, name)) != null) {
            return new MLP(
                    Double.parseDouble(params.get("ratio")),
                    Integer.parseInt(params.get("nLayers")),
                    params.containsKey("actFun") ? MultiLayerPerceptron.ActivationFunction.valueOf(params.get("actFun").toUpperCase()) : MultiLayerPerceptron.ActivationFunction.TANH
            );
        }
        if ((params = params(hlp, name)) != null) {
            if (params.get("type").equals("full")) {
                return new HLPFull(
                        Double.parseDouble(params.get("eta")),
                        params.containsKey("seed") ? new Random(Integer.parseInt(params.get("seed"))) : null,
                        params.containsKey("actFun") ? HebbianPerceptronFullModel.ActivationFunction.valueOf(params.get("actFun").toUpperCase()) : HebbianPerceptronFullModel.ActivationFunction.TANH,
                        Double.parseDouble(params.get("ratio")),
                        Integer.parseInt(params.get("nLayers")));
            } else {
                return new HLPOutput(
                        Double.parseDouble(params.get("eta")),
                        params.containsKey("seed") ? new Random(Integer.parseInt(params.get("seed"))) : null,
                        params.containsKey("actFun") ? HebbianPerceptronOutputModel.ActivationFunction.valueOf(params.get("actFun").toUpperCase()) : HebbianPerceptronOutputModel.ActivationFunction.TANH,
                        Double.parseDouble(params.get("ratio")),
                        Integer.parseInt(params.get("nLayers")));

            }

        }
        if ((params = params(pruningMlp, name)) != null) {
            return new PruningMLP(
                    Double.parseDouble(params.get("ratio")),
                    Integer.parseInt(params.get("nLayers")),
                    MultiLayerPerceptron.ActivationFunction.valueOf(params.get("actFun").toUpperCase()),
                    Double.parseDouble(params.get("pruningTime")),
                    Double.parseDouble(params.get("pruningRate")),
                    PruningMultiLayerPerceptron.Context.NETWORK,
                    PruningMultiLayerPerceptron.Criterion.valueOf(params.get("criterion").toUpperCase())

            );
        }
        if ((params = params(fgraph, name)) != null) {
            return new FGraph();
        }
        //misc
        if ((params = params(functionGrid, name)) != null) {
            return new FunctionGrid((PrototypedFunctionBuilder) getMapperBuilderFromName(params.get("innerMapper")));
        }
        if ((params = params(directNumGrid, name)) != null) {
            return new DirectNumbersGrid();
        }
        if ((params = params(functionNumGrid, name)) != null) {
            return new FunctionNumbersGrid();
        }
        throw new IllegalArgumentException(String.format("Unknown mapper name: %s", name));
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static Evolver<?, Robot<?>, Outcome> buildEvolver(String evolverName, String robotMapperName, Robot<?>
            target, Function<Outcome, Double> outcomeMeasure) {
        PrototypedFunctionBuilder<?, ?> mapperBuilder = null;
        for (String piece : robotMapperName.split(MAPPER_PIPE_CHAR)) {
            if (mapperBuilder == null) {
                mapperBuilder = getMapperBuilderFromName(piece);
            } else {
                mapperBuilder = mapperBuilder.compose((PrototypedFunctionBuilder) getMapperBuilderFromName(piece));
            }
        }
        return getEvolverBuilderFromName(evolverName).build(
                (PrototypedFunctionBuilder) mapperBuilder,
                target,
                PartialComparator.from(Double.class).comparing(outcomeMeasure).reversed()
        );
    }

    private static Function<Robot<?>, Outcome> buildTaskFromName(String transformationSequenceName, String
            terrainSequenceName, double episodeT, Random random) {
        //for sequence, assume format '99:name>99:name'd
        //transformations
        Function<Robot<?>, Robot<?>> transformation;
        if (transformationSequenceName.contains(SEQUENCE_SEPARATOR_CHAR)) {
            transformation = new SequentialFunction<>(getSequence(transformationSequenceName).entrySet().stream()
                    .collect(Collectors.toMap(
                            Map.Entry::getKey,
                            e -> RobotUtils.buildRobotTransformation(e.getValue(), random)
                            )
                    ));
        } else {
            transformation = RobotUtils.buildRobotTransformation(transformationSequenceName, random);
        }
        //terrains
        Function<Robot<?>, Outcome> task;
        if (terrainSequenceName.contains(SEQUENCE_SEPARATOR_CHAR)) {
            task = new SequentialFunction<>(getSequence(terrainSequenceName).entrySet().stream()
                    .collect(Collectors.toMap(
                            Map.Entry::getKey,
                            e -> buildLocomotionTask(e.getValue(), episodeT, random)
                            )
                    ));
        } else {
            task = buildLocomotionTask(terrainSequenceName, episodeT, random);
        }
        return task.compose(transformation);
    }

    public static Function<Robot<?>, Outcome> buildLocomotionTask(String terrainName, double episodeT, Random random) {
        if (!terrainName.contains("-rnd")) {
            return Misc.cached(new Locomotion(
                    episodeT,
                    Locomotion.createTerrain(terrainName),
                    PHYSICS_SETTINGS
            ), CACHE_SIZE);
        }
        return r -> new Locomotion(
                episodeT,
                Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
                PHYSICS_SETTINGS
        ).apply(r);
    }

    public static SortedMap<Long, String> getSequence(String sequenceName) {
        return new TreeMap<>(Arrays.stream(sequenceName.split(SEQUENCE_SEPARATOR_CHAR)).collect(Collectors.toMap(
                s -> s.contains(SEQUENCE_ITERATION_CHAR) ? Long.parseLong(s.split(SEQUENCE_ITERATION_CHAR)[0]) : 0,
                s -> s.contains(SEQUENCE_ITERATION_CHAR) ? s.split(SEQUENCE_ITERATION_CHAR)[1] : s
        )));
    }

}
