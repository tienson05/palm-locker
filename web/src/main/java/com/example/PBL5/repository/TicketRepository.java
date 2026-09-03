package com.example.PBL5.repository;

import com.example.PBL5.entity.Ticket;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface TicketRepository extends JpaRepository<Ticket, String> {
    Ticket findTopByOrderByIdDesc();
    @Query(value = "SELECT TO_CHAR(created_at, 'YYYY-MM-DD'), COUNT(*) FROM ticket WHERE created_at BETWEEN :start AND :end GROUP BY TO_CHAR(created_at, 'YYYY-MM-DD')", nativeQuery = true)
    List<Object[]> countTicketsByDay(@Param("start") java.time.LocalDateTime start, @Param("end") java.time.LocalDateTime end);

    @Query(value = "SELECT TO_CHAR(created_at, 'MM'), COUNT(*) FROM ticket WHERE created_at BETWEEN :start AND :end GROUP BY TO_CHAR(created_at, 'MM')", nativeQuery = true)
    List<Object[]> countTicketsByMonth(@Param("start") java.time.LocalDateTime start, @Param("end") java.time.LocalDateTime end);
}


