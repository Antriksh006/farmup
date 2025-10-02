package com.farmup.farmup_java_backend.repository;

import com.farmup.farmup_java_backend.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, String> {
    Optional<User> findByPhone(String phone);
    User getByPhone(String phone);
}
