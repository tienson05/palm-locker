package com.example.PBL5.service;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import com.example.PBL5.dto.adminOpenRequestDto;
import com.example.PBL5.dto.updateLocker;
import com.example.PBL5.entity.Locker;
import com.example.PBL5.entity.Session;
import com.example.PBL5.entity.Ticket;
import com.example.PBL5.repository.LockerRepository;
import com.example.PBL5.repository.SessionRepository;
import com.example.PBL5.repository.TicketRepository;
import com.example.PBL5.utils.IdGenerator;

@Service
public class LockerService {

    @Value("${esp32.ip}")
    private String ESP32_IP;
    @Autowired
    private SessionRepository sessionRepository;
    // Công cụ để Server gọi API sang con ESP32
    private final RestTemplate restTemplate = new RestTemplate();

    private final LockerRepository lockerRepository; // khai báo 1 biến trong class Service
    private final TicketRepository ticketRepository;
    public LockerService(LockerRepository lockerRepository, TicketRepository ticketRepository) {
        this.lockerRepository = lockerRepository;// gan obj cho bien cua class Service
        this.ticketRepository = ticketRepository;
    }
    public Locker createLocker(Locker locker) {
        if (lockerRepository.existsByLocation(locker.getLocation())) {
            throw new RuntimeException("Vị trí này đã có tủ đồ rồi!");
        }

        Locker lastLocker = lockerRepository.findTopByOrderByIdDesc();
        String lastId = null;
        if (lastLocker != null) {
            lastId = lastLocker.getId();
        }

        String newId = IdGenerator.generateId(lastId, "LK");
        locker.setId(newId);

        return lockerRepository.save(locker);
    }
    public List<Locker> getAllLockers() {
        return lockerRepository.findAll();
    }
    public Locker getLockerById(String id){
        return lockerRepository.findById(id).orElse(null);
    }

    public Locker updateLocker(String id, updateLocker request) {
        Locker locker = lockerRepository.findById(id).orElse(null);

        if (locker == null) {
            return null;
        }

        locker.setLocation(request.getLocation());
        locker.setStatus(request.getStatus());

        return lockerRepository.save(locker);

    }

    public void deleteLocker(String id) {
        Locker locker = lockerRepository.findById(id).orElse(null);

        if (locker == null) {
            return;
        }
        lockerRepository.delete(locker);
    }
    public List<Locker> searchLockers(String keyword, String status) {
        if (status == null || status.equals("ALL")) {
            return lockerRepository.searchAllStatus(keyword);
        }
        return lockerRepository.searchWithStatus(keyword, status);
    }



    public String adminOpenLocker(adminOpenRequestDto request) {
        try {
            String id = request.getLockerId();
            String reason = request.getReason();

            String doorNumber = String.valueOf(
                    Integer.parseInt(id.replaceAll("\\D+", ""))
            );
            String url = ESP32_IP + "/open?locker=" + doorNumber;

            String response = restTemplate.getForObject(url, String.class);

            if (response != null && response.contains("opened")) {
                Locker locker = lockerRepository.findById(id)
                        .orElseThrow(() -> new RuntimeException("Locker not found"));

                // 1. Cập nhật trạng thái tủ đồ về available dưới Database
                locker.setStatus("available");
                lockerRepository.save(locker);

                // 2. 🔥 ĐÃ SỬA: Bỏ .orElse(null) để khớp với hàm Repository trả về kiểu Session trực tiếp
                Session activeSession = sessionRepository.findByLockerIdAndStatus(id, "active");

                if (activeSession != null) {
                    activeSession.setStatus("inactive");

                    // ⚠️ TOÀN LƯU Ý CHỖ NÀY:
                    // Nếu trong Entity Session của bạn đặt tên biến thời gian kết thúc là end_time (viết thường),
                    // hãy đổi hàm dưới đây thành activeSession.setEnd_time(LocalDateTime.now()); cho khớp nhé!
                    activeSession.setEndTime(LocalDateTime.now());

                    sessionRepository.save(activeSession);
                }

                // 3. Khởi tạo Ticket báo cáo sự cố/mở khẩn cấp từ Ban quản lý
                Ticket lastTicket = ticketRepository.findTopByOrderByIdDesc();
                String lastId = null;
                if (lastTicket != null) {
                    lastId = lastTicket.getId();
                }

                String newId = IdGenerator.generateId(lastId, "TK");
                Ticket ticket = new Ticket();
                ticket.setId(newId);

                // Đồng bộ chuẩn tên trường created_at theo đúng Entity Ticket của Toàn
                ticket.setCreated_at(LocalDateTime.now());
                ticket.setReason(reason);
                ticket.setLocker(locker);

                // 🔥 ĐÃ ĐỒNG BỘ: Gắn kết trực tiếp Object thực thể Session vào Ticket (Không còn lỗi đỏ)
                if (activeSession != null) {
                    ticket.setSession(activeSession);
                }

                ticketRepository.save(ticket);
            }
            return "Locker " + id + " -> " + response;
        } catch (Exception e) {
            e.printStackTrace();
            throw new org.springframework.web.server.ResponseStatusException(
            org.springframework.http.HttpStatus.SERVICE_UNAVAILABLE, "Lỗi kết nối Server phần cứng hoặc AI offline: " + e.getMessage()
        
            );
        }
    }
    public Map<String, Object> getDashboardStatistics() {
        Map<String, Object> stats = new HashMap<>();

        // Đẩy toàn bộ gánh nặng tính toán nghiệp vụ xuống đây
        stats.put("total", lockerRepository.count());

        // 🔥 Bổ sung thêm "AVAILABLE" (In hoa) để quét sạch mọi trường hợp
        stats.put("free", lockerRepository.countByStatusIn(Arrays.asList("FREE", "available", "AVAILABLE")));

        // 🔥 Bổ sung thêm "OCCUPIED" (In hoa)
        stats.put("occupied", lockerRepository.countByStatusIn(Arrays.asList("occupied", "IN_USE", "OCCUPIED")));

        stats.put("error", lockerRepository.countByStatusIn(Arrays.asList("ERROR", "error")));        stats.put("supportCount", ticketRepository.count());

        return stats;
    }
    }

