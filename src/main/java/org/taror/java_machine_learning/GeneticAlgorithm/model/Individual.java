package org.taror.java_machine_learning.GeneticAlgorithm.model;

public class Individual {

    private int id;
    private double[] setOfChromosomes;
    private int age;
    private double result;

    public Individual(int id, double[] setOfChromosomes) {
        this.id = id;
        this.setOfChromosomes = setOfChromosomes;
        this.age = 0;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public double[] getSetOfChromosomes() {
        return setOfChromosomes;
    }

    public void setSetOfChromosomes(double[] setOfChromosomes) {
        this.setOfChromosomes = setOfChromosomes;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public double getResult() {
        return result;
    }

    public void setResult(double result) {
        this.result = result;
    }
}
