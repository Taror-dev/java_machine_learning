package org.taror.java_machine_learning.GeneticAlgorithm;

import org.taror.java_machine_learning.GeneticAlgorithm.model.Individual;
import org.taror.java_machine_learning.services.RandomNumber;

import java.util.ArrayList;
import java.util.List;

public class Mutation {

    public static List<Integer> RandomSelectionOfIndividualsForMutation(int numberOfMutatedIndividuals, int lengthPool) {

        List<Integer> listOfMutatedIndividuals = new ArrayList<>();

        for (int i = 0; i < numberOfMutatedIndividuals; i++) {

            int number = (int) Math.round(RandomNumber.randomInRange(0, lengthPool - 1));

            for (int j = 0; j < listOfMutatedIndividuals.size(); j++) {
                if (listOfMutatedIndividuals.get(j) == number) {
                    number = (int) Math.round(RandomNumber.randomInRange(0, lengthPool - 1));
                    j = 0;
                }
            }
            listOfMutatedIndividuals.add(number);
        }

        return listOfMutatedIndividuals;
    }

    public static Individual changingTheRandomChromosomeToTheMinimumValue(int id, Individual individual, int maxInputLayersCount, int maxHiddenLayersCount, int maxNumberOfNeuronsInTheHiddenLayer) {

        Individual mutatedIndividual = new Individual(id, new double[individual.getSetOfChromosomes().length]);

        for (int i = 0; i < individual.getSetOfChromosomes().length; i++) {
            mutatedIndividual.getSetOfChromosomes()[i] = individual.getSetOfChromosomes()[i];
        }

        int numbersOfMutatedChromosomes = (int) Math.round(RandomNumber.randomInRange(1, 6));//для мутации первой хромосомы поставить 0 вместо 1
        int number = (int) Math.round(RandomNumber.randomInRange(0, 1));
        int maxValue = 0;

        if (numbersOfMutatedChromosomes == 0 || numbersOfMutatedChromosomes == 1 || numbersOfMutatedChromosomes == 2) {

            if (numbersOfMutatedChromosomes == 0) {
                maxValue = maxInputLayersCount;
            } else if (numbersOfMutatedChromosomes == 1) {
                maxValue = maxHiddenLayersCount;
            } else {
                maxValue = maxNumberOfNeuronsInTheHiddenLayer;
            }

            if (mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] == maxValue) {
                mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] -= 1.0;
            } else if (mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] == 1.0) {
                mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] += 1.0;
            } else {
                if (number == 0) {
                    mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] -= 1.0;
                } else {
                    mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] += 1.0;
                }
            }
        } else {

            if (mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] == 1.0) {
                mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] -= 0.1;
            } else if (mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] == 0.0) {
                mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] += 0.1;
            } else {

                if (number == 0) {
                    if ((numbersOfMutatedChromosomes == 4 || numbersOfMutatedChromosomes == 5) &&
                            mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] == 0.1) {
                        mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] += 0.1;
                    } else {
                        mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] -= 0.1;
                    }
                } else if (number == 1) {
                    mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] += 0.1;
                }
            }
        }

        mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] =
                (double) Math.round(mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] * 10) / 10;

        return mutatedIndividual;
    }
}
