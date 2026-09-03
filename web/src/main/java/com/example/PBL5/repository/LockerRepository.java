package com.example.PBL5.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import com.example.PBL5.entity.Locker;

public interface LockerRepository extends JpaRepository<Locker, String> {
    Locker findTopByOrderByIdDesc();
    Locker findTopByStatus(String status);
    boolean existsByLocation(String location);
    @Query("SELECT l FROM Locker l WHERE (l.id LIKE %:keyword% OR l.location LIKE %:keyword%) AND l.status = :status")
    List<Locker> searchWithStatus(@Param("keyword") String keyword, @Param("status") String status);

    @Query("SELECT l FROM Locker l WHERE l.id LIKE %:keyword% OR l.location LIKE %:keyword%")
    List<Locker> searchAllStatus(@Param("keyword") String keyword);
    // Tự động sinh câu lệnh: SELECT COUNT(*) FROM locker WHERE status = ?
    long countByStatus(String status);

    // Tự động sinh câu lệnh: SELECT COUNT(*) FROM locker WHERE status IN (?, ?, ...)
    long countByStatusIn(List<String> statuses);
}