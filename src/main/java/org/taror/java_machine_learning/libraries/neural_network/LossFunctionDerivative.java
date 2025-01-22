package org.taror.java_machine_learning.libraries.neural_network;

public class LossFunctionDerivative {

    public static double mse(double output, double answer) {
        return 2 * (output - answer);
    }

    public static double crossEntropyPlusSoftmax(double output, double reward) {
        return output - reward;
    }
}
