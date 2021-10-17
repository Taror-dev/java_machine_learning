package org.taror.java_machine_learning.repositories;

import org.springframework.data.repository.CrudRepository;
import org.taror.java_machine_learning.entities.Solutions;

import java.util.List;

public interface SolutionsRepository extends CrudRepository<Solutions, Integer> {

    List<Solutions> findByOrderByIdAsc();
}
