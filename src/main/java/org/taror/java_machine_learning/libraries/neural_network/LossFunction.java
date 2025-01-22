package org.taror.java_machine_learning.libraries.neural_network;

public class LossFunction {

    public static double mse(double output, double answer) {

        return Math.pow(output - answer, 2);
    }

    public static double crossEntropy(double reward, double output) {

        return -1 * reward * Math.log(output);
    }
}
