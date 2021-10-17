package org.taror.java_machine_learning.entities;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;

@Entity
@Table(name = "solutions")
public class Solutions {

    @Id
    @Column(name = "id")
    private int id;

    @Column(name = "age")
    private int age;

    @Column(name = "set_of_chromosomes")
    private String setOfChromosomes;

    @Column(name = "result")
    private double result;

    public Solutions () {}

    public Solutions(int id, int age, String setOfChromosomes, double result) {
        this.id = id;
        this.age = age;
        this.setOfChromosomes = setOfChromosomes;
        this.result = result;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
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
}
