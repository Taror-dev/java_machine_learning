package org.taror.java_machine_learning.GeneticAlgorithm;

import org.taror.java_machine_learning.services.RandomNumber;

public class GenerateIndividual {

    public static double[] randomIndividual(int maxInputLayersCount, int maxHiddenLayersCount, int maximumNumberOfNeuronsInTheHiddenLayer) {

        double[] individual = new double[7];

        individual[0] = ((int) RandomNumber.randomInRange(1, maxInputLayersCount + 1));
        individual[1] = ((int) RandomNumber.randomInRange(1, maxHiddenLayersCount + 1));
        individual[2] = ((int) RandomNumber.randomInRange(1, maximumNumberOfNeuronsInTheHiddenLayer + 1));

        for (int i = 3; i < 7; i++) {
            if (i == 4 || i == 5) {
                individual[i] = (double) ((int) RandomNumber.randomInRange(1, 11)) / 10;
            } else {
                individual[i] = (double) ((int) RandomNumber.randomInRange(0, 11)) / 10;
            }
        }

        return individual;
    }
}
