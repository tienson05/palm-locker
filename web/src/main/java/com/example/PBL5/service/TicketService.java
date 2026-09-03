package com.example.PBL5.service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.example.PBL5.entity.Locker;
import com.example.PBL5.entity.Session;
import com.example.PBL5.entity.Ticket;
import com.example.PBL5.repository.LockerRepository;
import com.example.PBL5.repository.SessionRepository;
import com.example.PBL5.repository.TicketRepository;
import com.example.PBL5.utils.IdGenerator;
import com.example.PBL5.websocket.LockerServer;

@Service
public class TicketService {
@Autowired
private TicketRepository ticketRepository;
@Autowired
private SessionRepository sessionRepository;
@Autowired
private LockerRepository lockerRepository;
LockerServer lockerServer;

public void adminForceOpen(String lockerId, String reason) {
   Locker locker = lockerRepository.findById(lockerId)
           .orElseThrow(() -> new RuntimeException("Locker not found"));

   Session session = sessionRepository.findByLockerIdAndStatus(lockerId, "OCCUPIED");

   Ticket lastTicket = ticketRepository.findTopByOrderByIdDesc();

   String lastId = null;
   if (lastTicket != null) {
       lastId = lastTicket.getId();
   }

   String newId = IdGenerator.generateId(lastId, "TK");

   Ticket ticket = new Ticket();
   ticket.setId(newId);
   ticket.setLocker(locker);
   ticket.setSession(session);
   ticket.setReason(reason);
   ticket.setCreated_at(LocalDateTime.now());

   ticketRepository.save(ticket);
   lockerServer.openLocker(locker.getId());
}
// Thêm hàm này vào bên trong class TicketService của bạn:
public java.util.List<Ticket> getAllTickets() {
    return ticketRepository.findAll();
}
public List<Ticket> searchTickets(String keyword, String location, String startDate, String endDate) {
        // 1. Lấy toàn bộ danh sách Ticket từ Database lên
        List<Ticket> allTickets = ticketRepository.findAll(); 

        // 2. Dùng Stream để lọc chính xác theo yêu cầu mới
        return allTickets.stream()
                .filter(t -> {
                    // 🔹 Ô TÌM KIẾM 1: CHỈ LỌC THEO MÃ PHIÊN (SESSION ID)
                    if (keyword != null && !keyword.isEmpty()) {
                        String upperKey = keyword.toUpperCase().trim();
                        
                        // Lấy mã phiên (Nếu phiên sử dụng bị null thì trả về chuỗi rỗng)
                        String sessionId = t.getSession() != null ? String.valueOf(t.getSession().getId()).toUpperCase() : "";

                        // Nếu mã phiên của Ticket không chứa từ khóa tìm kiếm thì loại bỏ ngay
                        if (!sessionId.contains(upperKey)) {
                            return false;
                        }
                    }

                    // 🔹 Ô TÌM KIẾM 2: CHỈ LỌC THEO VỊ TRÍ TỦ
                    if (location != null && !location.isEmpty()) {
                        String upperLoc = location.toUpperCase().trim();
                        
                        // Lấy chuỗi vị trí từ thực thể Locker liên kết
                        String lockerLoc = t.getLocker() != null && t.getLocker().getLocation() != null 
                                           ? t.getLocker().getLocation().toUpperCase() : "";

                        // Nếu vị trí tủ đồ không chứa từ khóa thì loại bỏ
                        if (!lockerLoc.contains(upperLoc)) {
                            return false;
                        }
                    }

                    // 🔹 BỘ LỌC 3: LỌC THEO KHOẢNG THỜI GIAN (Hôm nay, Tháng, Năm, Custom)
                    if (startDate != null && !startDate.isEmpty() && endDate != null && !endDate.isEmpty()) {
                        LocalDateTime startDateTime = LocalDate.parse(startDate).atStartOfDay();
                        LocalDateTime endDateTime = LocalDate.parse(endDate).atTime(LocalTime.MAX);

                        LocalDateTime ticketTime = t.getCreated_at(); 
                        if (ticketTime == null) return false;

                        return !ticketTime.isBefore(startDateTime) && !ticketTime.isAfter(endDateTime);
                    }

                    return true; // Giữ lại Ticket nếu vượt qua hết các bộ lọc trên
                })
                .collect(Collectors.toList());
    }
}
