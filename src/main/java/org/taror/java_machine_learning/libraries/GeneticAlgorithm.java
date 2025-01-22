package org.taror.java_machine_learning.libraries;

import lombok.Data;
import org.taror.java_machine_learning.models.RestrictionsSettings;
import org.taror.java_machine_learning.models.Individual;

import java.util.Random;

@Data
public class GeneticAlgorithm {

    private RestrictionsSettings settings;
    private int id = 0;

    public Individual generateIndividual(int sid) {

        Individual individual = new Individual();

        individual.setId(incrementId());

        double[] chromosome = new double[8];

        chromosome[0] = ((int) randomInRange(1, settings.getMaxNumberOfInputs() + 1.0, sid));
        chromosome[1] = ((int) randomInRange(1, settings.getMaxNumberOfNeuronsInTheInputLayer() + 1.0, sid));
        chromosome[2] = ((int) randomInRange(1, settings.getMaxNumberOfHiddenLayers() + 1.0, sid));
        chromosome[3] = ((int) randomInRange(1, settings.getMaxNumberOfNeuronsInHiddenLayer() + 1.0, sid));
        chromosome[4] = (double) ((int) randomInRange(0, 11, sid)) / 10;
        chromosome[5] = (double) ((int) randomInRange(1, 11, sid)) / 10;
        chromosome[6] = (double) ((int) randomInRange(1, 11, sid)) / 10;
        chromosome[7] = (double) ((int) randomInRange(1, 11, sid)) / 10000;

        individual.setSetOfChromosomes(chromosome);
        individual.setAge(0);

        return individual;
    }

    public Individual mutate(Individual individual, int sid) {

        Individual mutatedIndividual = new Individual();
        mutatedIndividual.setId(incrementId());
        mutatedIndividual.setSetOfChromosomes(new double[individual.getSetOfChromosomes().length]);

        System.arraycopy(individual.getSetOfChromosomes(), 0, mutatedIndividual.getSetOfChromosomes(), 0, individual.getSetOfChromosomes().length);

        int numbersOfMutatedChromosomes = (int) Math.round(randomInRange(1, 7, sid));//для мутации первой хромосомы поставить 0 вместо 1
        int number = (int) Math.round(randomInRange(0, 1, sid));
        int maxValue;

        switch (numbersOfMutatedChromosomes) {
            case 0, 1, 2, 3 -> {

                switch (numbersOfMutatedChromosomes) {
                    case 0 -> maxValue = settings.getMaxNumberOfInputs();
                    case 1 -> maxValue = settings.getMaxNumberOfNeuronsInTheInputLayer();
                    case 2 -> maxValue = settings.getMaxNumberOfHiddenLayers();
                    default -> maxValue = settings.getMaxNumberOfNeuronsInHiddenLayer();
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
            }
            case 7 -> {

                if (mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] == 0.0001) {
                    mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] -= 0.00001;
                } else if (mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] == 0.00001) {
                    mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] += 0.00001;
                } else {
                    if (number == 0) {
                        mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] -= 0.00001;
                    } else if (number == 1) {
                        mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] += 0.00001;
                    }
                }

            }
            default -> {

                if (mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] == 1.0) {
                    mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] -= 0.1;
                } else if (mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] == 0.0) {
                    mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] += 0.1;
                } else {

                    if (number == 0) {
                        if ((numbersOfMutatedChromosomes == 5 || numbersOfMutatedChromosomes == 6) &&
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
        }

        mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] =
                (double) Math.round(mutatedIndividual.getSetOfChromosomes()[numbersOfMutatedChromosomes] * 10) / 10;

        return mutatedIndividual;
    }

    public Individual getDescendant(Individual first, Individual second, int sid) {

        double[] descendantsChromosome = new double[8];

        for (int i = 0; i < 8; i++) {

            int choice = (int) Math.round(randomInRange(0, 1, sid));

            if (choice == 0) {
                descendantsChromosome[i] = first.getSetOfChromosomes()[i];
            } else {
                descendantsChromosome[i] = second.getSetOfChromosomes()[i];
            }
        }

        Individual descendant = new Individual();
        descendant.setId(incrementId());
        descendant.setSetOfChromosomes(descendantsChromosome);
        descendant.setAge(0);

        return descendant;
    }

    public int incrementId() {
        id = id + 1;
        return id;
    }

    public double randomInRange(double start, double end, int sid) {

        Random random;

        if (sid == 0) {
            random = new Random();
        } else {
            random = new Random(sid);
        }

        return (random.nextDouble() * (end - start)) + start;
    }

    public void setRestrictionsSettings(RestrictionsSettings settings) {
        this.settings = settings;
    }
}
