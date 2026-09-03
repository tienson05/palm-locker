package com.example.PBL5.controller;

import java.util.List;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.example.PBL5.dto.adminOpenRequestDto;
import com.example.PBL5.entity.Ticket;
import com.example.PBL5.service.TicketService;

@RestController
@RequestMapping("/tickets") 
@CrossOrigin(origins = "*") // Cho phép Frontend JavaScript truy cập cấu trúc API
public class TicketController {
    
    private final TicketService ticketService;

    public TicketController(TicketService ticketService) {
        this.ticketService = ticketService;
    }

    // Endpoint lấy tất cả danh sách Ticket đổ ra bảng support.html
    @GetMapping
    public List<Ticket> getAllTickets() {
        return ticketService.getAllTickets();
    }

    @PostMapping("/force-open")
    public String forceOpen(@RequestBody adminOpenRequestDto request) {
        ticketService.adminForceOpen(
                request.getLockerId(),
                request.getReason()
        );
        return "locker opened";
    }
    @GetMapping("/search")
    public List<Ticket> searchTickets(
            @RequestParam(required = false) String keyword,
            @RequestParam(required = false) String location,
            @RequestParam(required = false) String startDate,
            @RequestParam(required = false) String endDate) {
            
        return ticketService.searchTickets(keyword, location, startDate, endDate);
    }
    
}