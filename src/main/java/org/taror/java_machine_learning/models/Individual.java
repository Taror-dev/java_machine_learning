package org.taror.java_machine_learning.models;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class Individual implements Comparable<Individual> {

    private int id;
    private double[] setOfChromosomes;
    private int age;
    private double result;

    private List<Double> referenceValues;
    private List<Double> resultValues;

    public Individual() {
        id = 0;
        setOfChromosomes = new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        age = 0;
        result = 1000.0;
        referenceValues = new ArrayList<>();
        resultValues = new ArrayList<>();
    }

    @Override
    public int compareTo(Individual individual) {
        return (int) (this.result * 1000 - individual.getResult() * 1000);
    }
}
