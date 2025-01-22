package org.taror.java_machine_learning;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.taror.java_machine_learning.models.RestrictionsSettings;
import org.taror.java_machine_learning.libraries.GeneticAlgorithm;
import org.taror.java_machine_learning.models.Individual;

class GeneticAlgorithmTests {

    private GeneticAlgorithm geneticAlgorithm;

    @BeforeEach
    public void preparation() {

        RestrictionsSettings restrictionsSettings = new RestrictionsSettings();
        restrictionsSettings.setId(1);
        restrictionsSettings.setIterationCounter(1000000);
        restrictionsSettings.setMaxNumberOfInputs(650);
        restrictionsSettings.setMaxNumberOfNeuronsInTheInputLayer(100);
        restrictionsSettings.setMaxNumberOfHiddenLayers(100);
        restrictionsSettings.setMaxNumberOfNeuronsInHiddenLayer(100);
        restrictionsSettings.setPoolLength(10);
        restrictionsSettings.setPercentageOfIndividualsForMutation(50);
        restrictionsSettings.setPercentageOfBestIndividualsInThePool(40);
        restrictionsSettings.setBestOfAllResult(1000);
        restrictionsSettings.setWorstOfAllResult(1000);

        geneticAlgorithm = new GeneticAlgorithm();
        geneticAlgorithm.setRestrictionsSettings(restrictionsSettings);
    }

    @Test
    void randomInRangeShouldBeCorrect() {

        double testValue = geneticAlgorithm.randomInRange(1, 10, 1);
        double result = 7.577903716329618;
        Assertions.assertEquals(result, testValue, 1e-9);
    }

    @Test
    void incrementIdShouldBeTwo() {

        double testValue = geneticAlgorithm.incrementId();
        double result = 1;
        Assertions.assertEquals(result, testValue, 1e-9);
    }

    @Test
    void newIndividualShouldBeCorrect() {

        Individual testIndividual = geneticAlgorithm.generateIndividual(1);

        Individual resultIndividual = new Individual();
        resultIndividual.setId(1);
        resultIndividual.setSetOfChromosomes(new double[] {476.0, 74.0, 74.0, 74.0, 0.8, 0.8, 0.8, 8.0E-4});
        resultIndividual.setAge(0);
        resultIndividual.setResult(0);

        Assertions.assertEquals(resultIndividual, testIndividual);
    }

    @Test
    void mutatedIndividualShouldBeCorrect() {

        Individual individual = geneticAlgorithm.generateIndividual(1);
        Individual mutatedIndividual = geneticAlgorithm.mutate(individual, 1);

        Individual resultIndividual = new Individual();
        resultIndividual.setId(2);
        resultIndividual.setSetOfChromosomes(new double[] {476.0, 74.0, 74.0, 74.0, 0.8, 0.9, 0.8, 8.0E-4});
        resultIndividual.setAge(0);
        resultIndividual.setResult(0);

        Assertions.assertEquals(resultIndividual, mutatedIndividual);
    }

    @Test
    void newDescendantShouldBeCorrect() {

        Individual firstIndividual = geneticAlgorithm.generateIndividual(1654967984);
        Individual secondIndividual = geneticAlgorithm.generateIndividual(468796497);

        Individual descendant = geneticAlgorithm.getDescendant(firstIndividual, secondIndividual, 1);

        Individual resultIndividual = new Individual();
        resultIndividual.setId(3);
        resultIndividual.setSetOfChromosomes(new double[] {505.0, 78.0, 78.0, 78.0, 0.8, 0.8, 0.8, 8.0E-4});
        resultIndividual.setAge(0);
        resultIndividual.setResult(0);

        Assertions.assertEquals(resultIndividual, descendant);
    }
}
