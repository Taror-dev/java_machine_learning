package org.taror.java_machine_learning.services;

public class RandomNumber {

    public static double randomInRange(double start, double end) {
        return (Math.random() * (end - start)) + start;
    }
}
