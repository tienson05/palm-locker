package com.example.PBL5.service;

import com.example.PBL5.dto.DashboardTrendDto;
import com.example.PBL5.repository.SessionRepository;
import com.example.PBL5.repository.TicketRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
@Service
public class TrendService {
    @Autowired
    private SessionRepository sessionRepository;

    @Autowired
    private TicketRepository ticketRepository;

    public DashboardTrendDto getTrendData(String type, LocalDate startDate, LocalDate endDate) {
        LocalDateTime startDateTime;
        LocalDateTime endDateTime = LocalDateTime.now();

        List<String> labels = new ArrayList<>();
        Map<String, Long> usageMap = new LinkedHashMap<>();
        Map<String, Long> ticketsMap = new LinkedHashMap<>();

        if ("year".equalsIgnoreCase(type)) {
            startDateTime = LocalDateTime.of(LocalDate.now().withDayOfYear(1), LocalTime.MIN);
            endDateTime = LocalDateTime.of(LocalDate.now().withDayOfYear(LocalDate.now().lengthOfYear()), LocalTime.MAX);

            for (int i = 1; i <= 12; i++) {
                String monthKey = String.format("%02d", i);
                labels.add("Tháng " + monthKey);
                usageMap.put(monthKey, 0L);
                ticketsMap.put(monthKey, 0L);
            }

            fillData(usageMap, sessionRepository.countUsageByMonth(startDateTime, endDateTime));
            fillData(ticketsMap, ticketRepository.countTicketsByMonth(startDateTime, endDateTime));

        } else {
            if ("week".equalsIgnoreCase(type)) {
                startDateTime = LocalDateTime.of(LocalDate.now().minusDays(6), LocalTime.MIN);
            } else if ("month".equalsIgnoreCase(type)) {
                startDateTime = LocalDateTime.of(LocalDate.now().withDayOfMonth(1), LocalTime.MIN);
            } else if ("custom".equalsIgnoreCase(type) && startDate != null && endDate != null) {
                startDateTime = LocalDateTime.of(startDate, LocalTime.MIN);
                endDateTime = LocalDateTime.of(endDate, LocalTime.MAX);
            } else {
                startDateTime = LocalDateTime.of(LocalDate.now().minusDays(6), LocalTime.MIN);
            }

            LocalDate tempDate = startDateTime.toLocalDate();
            LocalDate lastDate = endDateTime.toLocalDate();
            DateTimeFormatter displayFmt = DateTimeFormatter.ofPattern("dd/MM");
            DateTimeFormatter dbFmt = DateTimeFormatter.ofPattern("yyyy-MM-dd");

            while (!tempDate.isAfter(lastDate)) {
                String dbKey = tempDate.format(dbFmt);
                labels.add(tempDate.format(displayFmt));
                usageMap.put(dbKey, 0L);
                ticketsMap.put(dbKey, 0L);
                tempDate = tempDate.plusDays(1);
            }

            fillData(usageMap, sessionRepository.countUsageByDay(startDateTime, endDateTime));
            fillData(ticketsMap, ticketRepository.countTicketsByDay(startDateTime, endDateTime));
        }

        return new DashboardTrendDto(
                labels,
                new ArrayList<>(usageMap.values()),
                new ArrayList<>(ticketsMap.values())
        );
    }

    private void fillData(Map<String, Long> map, List<Object[]> rawData) {
        if (rawData != null) {
            for (Object[] row : rawData) {
                String key = String.valueOf(row[0]);
                if (map.containsKey(key)) {
                    map.put(key, Long.parseLong(String.valueOf(row[1])));
                }
            }
        }
    }
}

