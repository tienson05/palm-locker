package com.example.PBL5.dto;

import java.util.List;

public class DashboardTrendDto {
    private List<String> labels;
    private List<Long> usage;
    private List<Long> tickets;

    public DashboardTrendDto(List<String> labels, List<Long> usage, List<Long> tickets) {
        this.labels = labels;
        this.usage = usage;
        this.tickets = tickets;
    }

    public List<String> getLabels() {
        return labels;
    }

    public List<Long> getUsage() {
        return usage;
    }

    public List<Long> getTickets() {
        return tickets;
    }
}


