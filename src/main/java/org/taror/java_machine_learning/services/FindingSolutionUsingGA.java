package org.taror.java_machine_learning.services;

import jakarta.annotation.PostConstruct;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.taror.java_machine_learning.configuration.AppConfig;
import org.taror.java_machine_learning.libraries.GeneticAlgorithm;
import org.taror.java_machine_learning.libraries.neural_network.models.ActFun;
import org.taror.java_machine_learning.libraries.neural_network.models.LossFun;
import org.taror.java_machine_learning.models.Individual;
import org.taror.java_machine_learning.models.RestrictionsSettings;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

@Slf4j
@Service
@RequiredArgsConstructor
public class FindingSolutionUsingGA {

    private final AppConfig config;

    private GeneticAlgorithm geneticAlgorithm;

    private int processors;
    @Getter
    private Individual theBestIndividual;

    @Getter
    @Setter
    private boolean trainingStarted = false;

    @PostConstruct
    private void preparing() {

        geneticAlgorithm = new GeneticAlgorithm();

        RestrictionsSettings restrictionsSettings = new RestrictionsSettings();
        restrictionsSettings.setMaxNumberOfInputs(40);
        restrictionsSettings.setMaxNumberOfNeuronsInTheInputLayer(10);
        restrictionsSettings.setMaxNumberOfHiddenLayers(10);
        restrictionsSettings.setMaxNumberOfNeuronsInHiddenLayer(10);
        restrictionsSettings.setPoolLength(100);
        restrictionsSettings.setPercentageOfIndividualsForMutation(60);
        restrictionsSettings.setPercentageOfBestIndividualsInThePool(40);

        geneticAlgorithm.setRestrictionsSettings(restrictionsSettings);
        theBestIndividual = new Individual();

        processors = config.getNumberOfProcessors();

        if (processors == 0) {

            processors = Runtime.getRuntime().availableProcessors();

            if (processors > 4) {
                processors -= 2;
            }
        }
    }

    public void run() throws InterruptedException {

        trainingStarted = true;

        Set<Individual> buffer = new HashSet<>();

        generationOfIndividuals(buffer);
        evaluateOfIndividuals(buffer);
        selection(buffer);

        while (trainingStarted) {

            generationOfIndividuals(buffer);
            mutateIndividuals(buffer, 0);
            crossbreeding(buffer, 0);
            evaluateOfIndividuals(buffer);
            selection(buffer);

            if (theBestIndividual.getResult() < 0.005) {
                trainingStarted = false;
                log.info("Solution found {}", theBestIndividual);
                break;
            }
        }
    }

    private void generationOfIndividuals(Set<Individual> buffer) {

        for (int i = buffer.size(); i < geneticAlgorithm.getSettings().getPoolLength(); i++) {
            Individual individual = geneticAlgorithm.generateIndividual(0);
            individual.getSetOfChromosomes()[0] = 40;
            individual.getSetOfChromosomes()[1] = 1;
            buffer.add(individual);
        }
    }

    private void mutateIndividuals(Set<Individual> buffer, int sid) {

        Set<Individual> mutatedIndividuals = new HashSet<>();

        for (Individual individual : buffer) {
            mutatedIndividuals.add(geneticAlgorithm.mutate(individual, sid));
        }

        buffer.addAll(mutatedIndividuals);
    }

    private void crossbreeding(Set<Individual> buffer, int sid) {

        List<Individual> copyOfBuffer = new ArrayList<>(buffer);

        while (!copyOfBuffer.isEmpty()) {

            if (copyOfBuffer.size() == 1) {
                copyOfBuffer.removeFirst();
                break;
            }

            int individualNumber = (int) Math.round(geneticAlgorithm.randomInRange(0, copyOfBuffer.size() - 1.0, sid));
            Individual firstParent = copyOfBuffer.get(individualNumber);
            copyOfBuffer.remove(individualNumber);
            individualNumber = (int) Math.round(geneticAlgorithm.randomInRange(0, copyOfBuffer.size() - 1.0, sid));
            Individual secondParent = copyOfBuffer.get(individualNumber);
            copyOfBuffer.remove(individualNumber);

            buffer.add(geneticAlgorithm.getDescendant(firstParent, secondParent, sid));
        }
    }

    private void evaluateOfIndividuals(Set<Individual> buffer) throws InterruptedException {

        try (ExecutorService threadPool = Executors.newFixedThreadPool(processors)) {
            for (Individual individual : buffer) {

                threadPool.submit(
                        new Thread(() -> {
                            Individual tempIndividual = getResultFromNeuralNetwork(individual);
                            individual.setResult(tempIndividual.getResult());
                            individual.setReferenceValues(tempIndividual.getReferenceValues());
                            individual.setResultValues(tempIndividual.getResultValues());
                        }));
            }

            threadPool.shutdown();
            threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
        }
    }

    private void selection(Set<Individual> buffer) {

        TreeSet<Individual> sorted = new TreeSet<>(buffer);
        TreeSet<Individual> sorted2 = new TreeSet<>();

        for (Individual individual : sorted) {
            HashSet<Double> tempValues = new HashSet<>(individual.getResultValues());
            if (tempValues.size() != 1) {
                sorted2.add(individual);
            }
        }

        sorted.clear();

        if (theBestIndividual.getResult() > sorted2.getFirst().getResult()) {
            theBestIndividual = sorted2.getFirst();
            log.info("New best individual: {}", theBestIndividual);
        }

        buffer.clear();

        int counter = 0;

        for (Individual individual : sorted2) {
            if (counter < (geneticAlgorithm.getSettings().getPercentageOfBestIndividualsInThePool() * geneticAlgorithm.getSettings().getPoolLength()) / 100) {
                individual.setAge(individual.getAge() + 1);
                buffer.add(individual);
                ++counter;
            } else {
                break;
            }
        }
        sorted2.clear();
    }

    private Individual getResultFromNeuralNetwork(Individual individual) {

        NNSinX network = new NNSinX(individual, ActFun.TANH, LossFun.MSE);
        network.run();
        individual.setResult(network.getError());
        individual.setReferenceValues(network.getAnswers());
        individual.setResultValues(network.getResults());

        return individual;
    }
}
