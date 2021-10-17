package org.taror.java_machine_learning.entities;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;
import java.util.Arrays;

@Entity
@Table(name = "the_best_solution")
public class TheBestSolution {

    @Id
    @Column(name = "id")
    private int id;

    @Column(name = "result")
    private double result;

    @Column(name = "set_of_chromosomes")
    private String setOfChromosomes;

    @Column(name = "weights")
    private String weights;

    public TheBestSolution() {}

    public TheBestSolution(int id, double result, String setOfChromosomes, String weights) {
        this.id = id;
        this.result = result;
        this.setOfChromosomes = setOfChromosomes;
        this.weights = weights;
    }

    public TheBestSolution(int id, double result, double[] setOfChromosomes, double[][][] weights) {
        this.id = id;
        this.result = result;
        this.setOfChromosomes = Arrays.toString(setOfChromosomes);
        this.weights = Arrays.deepToString(weights);
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getSetOfChromosomes() {
        return setOfChromosomes;
    }

    public void setSetOfChromosomes(String setOfChromosomes) {
        this.setOfChromosomes = setOfChromosomes;
    }

    public double getResult() {
        return result;
    }

    public void setResult(double result) {
        this.result = result;
    }

    public String getWeights() {
        return weights;
    }

    public void setWeights(String weights) {
        this.weights = weights;
    }
}
