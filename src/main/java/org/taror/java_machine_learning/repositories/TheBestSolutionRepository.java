package org.taror.java_machine_learning.repositories;

import org.springframework.data.repository.CrudRepository;
import org.taror.java_machine_learning.entities.TheBestSolution;

import java.util.List;

public interface TheBestSolutionRepository  extends CrudRepository<TheBestSolution, Integer> {

    List<TheBestSolution> findByOrderByIdAsc();

}
