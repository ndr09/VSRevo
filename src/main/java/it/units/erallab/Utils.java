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

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.listener.Accumulator;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.listener.NamedFunctions;
import it.units.malelab.jgea.core.listener.TableBuilder;
import it.units.malelab.jgea.core.util.*;
import org.apache.commons.math3.util.Precision;
import org.dyn4j.dynamics.Settings;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static it.units.malelab.jgea.core.listener.NamedFunctions.*;

/**
 * @author eric
 */
public class Utils {

    private static final Logger L = Logger.getLogger(Utils.class.getName());

    private Utils() {
    }

    public static List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> keysFunctions() {
        return List.of(
                eventAttribute("experiment.name"),
                eventAttribute("seed", "%2d"),
                eventAttribute("terrain"),
                eventAttribute("shape"),
                eventAttribute("sensor.config"),
                eventAttribute("mapper"),
                eventAttribute("transformation"),
                eventAttribute("evolver")
        );
    }

    public static List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> basicFunctions() {
        return List.of(
                iterations(),
                births(),
                fitnessEvaluations(),
                elapsedSeconds()
        );
    }

    public static List<NamedFunction<Individual<?, ? extends Robot<?>, ? extends Outcome>, ?>> serializationFunction(boolean flag) {
        if (!flag) {
            return List.of();
        }
        return List.of(f("serialized", r -> SerializationUtils.serialize(r, SerializationUtils.Mode.GZIPPED_JSON)).of(solution()));
    }

    @SuppressWarnings("unchecked")
    public static List<NamedFunction<Individual<?, ? extends Robot<?>, ? extends Outcome>, ?>> individualFunctions(Function<Outcome, Double> fitnessFunction) {
        NamedFunction<Individual<?, ? extends Robot<?>, ? extends Outcome>, ?> size = size().of(genotype());
        NamedFunction<Outcome, String> mapx = o -> {
            Outcome.Gait gait = o.getMainGait();
            double[] x;
            if (gait == null) {
                x = new double[]{0d,0d};
            }else {
                x = new double[]{
                        gait.getAvgTouchArea(),
                        gait.getCoverage()
                };
            }

            return Arrays.toString(x).replace(" ", "");
        };
        NamedFunction<Outcome, String> mapy = o -> {
            Outcome.Gait gait = o.getMainGait();
            double[] y;
            if (gait == null) {
                y = new double[]{0d,0d};
            }else {
                y = new double[]{
                        gait.getPurity(),
                        gait.getModeInterval()

                };
            }
            return Arrays.toString(y).replace(" ", "");
        };

        return List.of(
                f("w", "%2d", (Function<Grid<?>, Number>) Grid::getW)
                        .of(f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels))
                        .of(solution()),
                f("h", "%2d", (Function<Grid<?>, Number>) Grid::getH)
                        .of(f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels))
                        .of(solution()),
                f("num.voxel", "%2d", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull))
                        .of(f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels))
                        .of(solution()),
                size.reformat("%5d"),
                genotypeBirthIteration(),
                f("mapElites.x", "%s", mapx).of(fitness()),
                f("mapElites.y", "%s", mapy).of(fitness()),
                f("fitness", "%5.1f", fitnessFunction).of(fitness())
        );
    }

    public static List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> populationFunctions(Function<Outcome, Double> fitnessFunction) {
        NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?> min = min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all());
        NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?> median = median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all());
        return List.of(
                size().of(all()),
                size().of(firsts()),
                size().of(lasts()),
                uniqueness().of(each(genotype())).of(all()),
                uniqueness().of(each(solution())).of(all()),
                uniqueness().of(each(fitness())).of(all()),
                min.reformat("%+4.1f"),
                median.reformat("%5.1f")
        );
    }

    public static List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> visualFunctions(Function<Outcome, Double> fitnessFunction) {
        return List.of(
                hist(8)
                        .of(each(f("fitness", fitnessFunction).of(fitness())))
                        .of(all()),
                hist(8)
                        .of(each(f("num.voxels", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull))
                                .of(f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels))
                                .of(solution())))
                        .of(all()),
                f("minimap", "%4s", (Function<Grid<?>, String>) g -> TextPlotter.binaryMap(
                        g.toArray(Objects::nonNull),
                        (int) Math.min(Math.ceil((float) g.getW() / (float) g.getH() * 2f), 4)))
                        .of(f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels))
                        .of(solution()).of(best()),
                f("average.posture.minimap", "%2s", (Function<Outcome, String>) o -> TextPlotter.binaryMap(o.getAveragePosture().toArray(b -> b), 2))
                        .of(fitness()).of(best())
        );
    }

    public static List<NamedFunction<Outcome, ?>> basicOutcomeFunctions() {
        return List.of(
                f("computation.time", "%4.2f", Outcome::getComputationTime),
                f("distance", "%5.1f", Outcome::getDistance),
                f("velocity", "%5.1f", Outcome::getVelocity),
                f("corrected.efficiency", "%5.2f", Outcome::getCorrectedEfficiency),
                f("area.ratio.power", "%5.1f", Outcome::getAreaRatioPower),
                f("control.power", "%5.1f", Outcome::getControlPower)
        );
    }

    public static List<NamedFunction<Outcome, ?>> mapElitesXY(double spectrumMinFreq, double spectrumMaxFreq, int spectrumSize) {
        return Misc.concat(List.of(NamedFunction.then(cachedF("mapELite.x",
                o -> o.getCenterPowerSpectrum(Outcome.Component.X, spectrumMinFreq, spectrumMaxFreq, spectrumSize).stream()
                        .map(Outcome.Mode::getStrength)
                        .collect(Collectors.toList())),
                IntStream.range(0, spectrumSize).mapToObj(NamedFunctions::nth).collect(Collectors.toList())
                ),
                NamedFunction.then(cachedF("mapELite.y",
                        o -> o.getCenterPowerSpectrum(Outcome.Component.Y, spectrumMinFreq, spectrumMaxFreq, spectrumSize).stream()
                                .map(Outcome.Mode::getStrength)
                                .collect(Collectors.toList())),
                        IntStream.range(0, spectrumSize).mapToObj(NamedFunctions::nth).collect(Collectors.toList())
                )
        ));
    }

    public static List<NamedFunction<Outcome, ?>> detailedOutcomeFunctions(double spectrumMinFreq, double spectrumMaxFreq, int spectrumSize) {
        return Misc.concat(List.of(
                NamedFunction.then(cachedF("gait", Outcome::getMainGait), List.of(
                        f("avg.touch.area", "%4.2f", g -> g == null ? null : g.getAvgTouchArea()),
                        f("coverage", "%4.2f", g -> g == null ? null : g.getCoverage()),
                        f("num.footprints", "%2d", g -> g == null ? null : g.getFootprints().size()),
                        f("mode.interval", "%3.1f", g -> g == null ? null : g.getModeInterval()),
                        f("purity", "%4.2f", g -> g == null ? null : g.getPurity()),
                        f("num.unique.footprints", "%2d", g -> g == null ? null : g.getFootprints().stream().distinct().count()),
                        f("footprints", g -> g == null ? null : g.getFootprints().stream().map(Objects::toString).collect(Collectors.joining(",")))
                )),
                NamedFunction.then(cachedF("center.spectrum.x",
                        o -> o.getCenterPowerSpectrum(Outcome.Component.X, spectrumMinFreq, spectrumMaxFreq, spectrumSize).stream()
                                .map(Outcome.Mode::getStrength)
                                .collect(Collectors.toList())),
                        IntStream.range(0, spectrumSize).mapToObj(NamedFunctions::nth).collect(Collectors.toList())
                ),
                NamedFunction.then(cachedF("center.spectrum.y",
                        o -> o.getCenterPowerSpectrum(Outcome.Component.Y, spectrumMinFreq, spectrumMaxFreq, spectrumSize).stream()
                                .map(Outcome.Mode::getStrength)
                                .collect(Collectors.toList())),
                        IntStream.range(0, spectrumSize).mapToObj(NamedFunctions::nth).collect(Collectors.toList())
                )
        ));
    }

    public static List<NamedFunction<Outcome, ?>> visualOutcomeFunctions(double spectrumMinFreq, double spectrumMaxFreq) {
        return List.of(
                cachedF("center.spectrum.x", "%8.8s", o -> TextPlotter.barplot(
                        o.getCenterPowerSpectrum(Outcome.Component.X, spectrumMinFreq, spectrumMaxFreq, 2).stream()
                                .mapToDouble(Outcome.Mode::getStrength)
                                .toArray())),
                cachedF("center.spectrum.y", "%8.8s", o -> TextPlotter.barplot(
                        o.getCenterPowerSpectrum(Outcome.Component.Y, spectrumMinFreq, spectrumMaxFreq, 2).stream()
                                .mapToDouble(Outcome.Mode::getStrength)
                                .toArray()))
        );
    }

    public static Accumulator.Factory<Event<?, ? extends Robot<?>, ? extends Outcome>, String> lastEventToString(Function<Outcome, Double> fitnessFunction) {
        final List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> functions = Misc.concat(List.of(
                keysFunctions(),
                basicFunctions(),
                populationFunctions(fitnessFunction),
                NamedFunction.then(best(), individualFunctions(fitnessFunction)),
                NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), basicOutcomeFunctions())
        ));
        return Accumulator.Factory.<Event<?, ? extends Robot<?>, ? extends Outcome>>last().then(
                e -> functions.stream()
                        .map(f -> f.getName() + ": " + f.applyAndFormat(e))
                        .collect(Collectors.joining("\n"))
        );
    }

    public static Accumulator.Factory<Event<?, ? extends Robot<?>, ? extends Outcome>, BufferedImage> fitnessPlot(Function<Outcome, Double> fitnessFunction) {
        return new TableBuilder<Event<?, ? extends Robot<?>, ? extends Outcome>, Number>(List.of(
                iterations(),
                f("fitness", fitnessFunction).of(fitness()).of(best()),
                min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all()),
                median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all())
        )).then(ImagePlotters.xyLines(600, 400));
    }

    public static Accumulator.Factory<Event<?, ? extends Robot<?>, ? extends Outcome>, BufferedImage> centerPositionPlot() {
        return Accumulator.Factory.<Event<?, ? extends Robot<?>, ? extends Outcome>>last().then(
                event -> {
                    Outcome o = Misc.first(event.getOrderedPopulation().firsts()).getFitness();
                    Table<Number> table = new ArrayTable<>(List.of("x", "y", "terrain.y"));
                    o.getObservations().forEach(obs -> table.addRow(List.of(
                            obs.getCenterPosition().x,
                            obs.getCenterPosition().y,
                            obs.getTerrainHeight()
                    )));
                    return table;
                }
        ).then(ImagePlotters.xyLines(600, 400));
    }

    public static Accumulator.Factory<Event<?, ? extends Robot<?>, ? extends Outcome>, File> bestVideo(double transientTime, double episodeTime, Settings settings) {
        return Accumulator.Factory.<Event<?, ? extends Robot<?>, ? extends Outcome>>last().then(
                event -> {
                    Random random = new Random(0);
                    SortedMap<Long, String> terrainSequence = LocomotionEvolution.getSequence((String) event.getAttributes().get("terrain"));
                    SortedMap<Long, String> transformationSequence = LocomotionEvolution.getSequence((String) event.getAttributes().get("transformation"));
                    String terrainName = terrainSequence.get(terrainSequence.lastKey());
                    String transformationName = transformationSequence.get(transformationSequence.lastKey());
                    Robot<?> robot = SerializationUtils.clone(Misc.first(event.getOrderedPopulation().firsts()).getSolution());
                    robot = RobotUtils.buildRobotTransformation(transformationName, random).apply(robot);
                    Locomotion locomotion = new Locomotion(
                            episodeTime,
                            Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
                            settings
                    );
                    File file;
                    try {
                        file = File.createTempFile("robot-video", ".mp4");
                        GridFileWriter.save(locomotion, robot, 300, 200, transientTime, 25, VideoUtils.EncoderFacility.JCODEC, file);
                        file.deleteOnExit();
                    } catch (IOException ioException) {
                        L.warning(String.format("Cannot save video of best: %s", ioException));
                        return null;
                    }
                    return file;
                }
        );
    }

    public static Function<Event<?, ? extends Robot<?>, ? extends Outcome>, Collection<LocomotionEvolution.ValidationOutcome>> validation(
            List<String> validationTerrainNames,
            List<String> validationTransformationNames,
            List<Integer> seeds,
            double episodeTime
    ) {
        return event -> {
            Robot<?> robot = SerializationUtils.clone(Misc.first(event.getOrderedPopulation().firsts()).getSolution());
            List<LocomotionEvolution.ValidationOutcome> validationOutcomes = new ArrayList<>();
            for (String validationTerrainName : validationTerrainNames) {
                for (String validationTransformationName : validationTransformationNames) {
                    for (int seed : seeds) {
                        Random random = new Random(seed);
                        robot = RobotUtils.buildRobotTransformation(validationTransformationName, random).apply(robot);
                        Function<Robot<?>, Outcome> validationLocomotion = LocomotionEvolution.buildLocomotionTask(
                                validationTerrainName,
                                episodeTime,
                                random
                        );
                        Outcome outcome = validationLocomotion.apply(robot);
                        validationOutcomes.add(new LocomotionEvolution.ValidationOutcome(
                                event,
                                Map.ofEntries(
                                        Map.entry("validation.terrain", validationTerrainName),
                                        Map.entry("validation.transformation", validationTransformationName),
                                        Map.entry("validation.seed", seed)
                                ),
                                outcome
                        ));
                    }
                }
            }
            return validationOutcomes;
        };
    }
}
