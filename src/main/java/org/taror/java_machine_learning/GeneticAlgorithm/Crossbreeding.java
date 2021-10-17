package org.taror.java_machine_learning.GeneticAlgorithm;

import org.taror.java_machine_learning.GeneticAlgorithm.model.Individual;
import org.taror.java_machine_learning.services.RandomNumber;

import java.util.List;

public class Crossbreeding {

    public static Individual oneDescendantRandomGenes(int id, List<Individual> selectedPairsForCrossing) {

        double[] dependent = new double[selectedPairsForCrossing.get(0).getSetOfChromosomes().length];

        for (int i = 0; i < selectedPairsForCrossing.get(0).getSetOfChromosomes().length; i++) {
            int choice = (int) Math.round(RandomNumber.randomInRange(0, 1));
            dependent[i] = selectedPairsForCrossing.get(choice).getSetOfChromosomes()[i];
        }

        return new Individual(id, dependent);
    }
}
