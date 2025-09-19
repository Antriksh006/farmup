package com.farmup.farmup_java_backend.repository;

import com.farmup.farmup_java_backend.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository extends JpaRepository<User, String> {
    User findByPhone(String phone);
}
