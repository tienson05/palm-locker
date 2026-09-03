package com.example.PBL5.service;
import java.io.File;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;

import com.example.PBL5.dto.SessionResponse;
import com.example.PBL5.entity.Session;
import com.example.PBL5.repository.SessionRepository;

@Service
public class SessionService {
    private final SessionRepository sessionRepository; // khai báo 1 biến trong class Service
    private final String STORAGE_PATH = "D:/Projects/Personal/PalmLocker/storage";

    public SessionService(SessionRepository sessionRepository) {
        this.sessionRepository = sessionRepository;// gan obj cho bien cua class Service
    }

    private SessionResponse convertToDTO(Session session) {
    SessionResponse response = new SessionResponse();
    response.setId(session.getId());

    // Gọi các getter mới
    response.setStartTime(session.getStartTime());
    response.setEndTime (session.getEndTime());
    response.setStatus(session.getStatus());
    response.setLockerLocation(session.getLocker() != null ? session.getLocker().getLocation() : null);

    if (session.getLocker() != null) {
        response.setLockerId(session.getLocker().getId());
    }
    return response;
}

   public List<SessionResponse> getAllSessions() {
    List<Session> sessions = sessionRepository.findAll();
    List<SessionResponse> responseList = new java.util.ArrayList<>();

    for (Session session : sessions) {
        SessionResponse res = new SessionResponse();
        res.setId(session.getId());

        if (session.getLocker() != null) {
            res.setLockerId(session.getLocker().getId());
            res.setLockerLocation(session.getLocker().getLocation()); // 🔥 Nạp vị trí tủ
        }

        res.setStartTime(session.getStartTime());
        res.setEndTime(session.getEndTime());
        res.setStatus(session.getStatus());
        responseList.add(res);
    }
    return responseList;
}

    public SessionResponse getSessionById(String id) {
        Session session = sessionRepository.findById(id).orElse(null);

        if (session == null) {
            return null;
        }

        SessionResponse response = convertToDTO(session);
        return response;
    }

    public SessionResponse getCurrentSession(String lockerId) {
        Session session = sessionRepository.findByLockerIdAndStatus(lockerId, "active");
        if (session == null) {
            return null;
        }
        return convertToDTO(session);
    }


public List<SessionResponse> searchSessions(String lockerId, String status, String sortBy, String startDate, String endDate) {
        // sortBy gửi lên sẽ là "start_time" hoặc "end_time", khớp 100% với Entity
        String fieldName = sortBy;
        if ("start_time".equals(sortBy) || "startTime".equals(sortBy)) fieldName = "startTime";
        if ("end_time".equals(sortBy) || "endTime".equals(sortBy)) fieldName = "endTime";

        Sort sort = Sort.by(Sort.Direction.ASC, fieldName);

        List<Session> sessions;
        boolean hasLockerId = lockerId != null && !lockerId.isEmpty();
        boolean hasStatus = status != null && !status.equalsIgnoreCase("ALL");

        // 1. Giữ nguyên logic lấy dữ liệu nền tảng từ database của bạn
        if (hasLockerId && hasStatus) {
            sessions = sessionRepository.findByLockerIdAndStatus(lockerId, status, sort);
        } else if (hasLockerId) {
            sessions = sessionRepository.findByLockerId(lockerId, sort);
        } else if (hasStatus) {
            sessions = sessionRepository.findByStatus(status, sort);
        } else {
            sessions = sessionRepository.findAll(sort);
        }

        // 2. 🔥 BỔ SUNG LOGIC LỌC THỜI GIAN THEO KỊCH BẢN (NGÀY, THÁNG, NĂM, CUSTOM)
        // Nếu Front-end có truyền khoảng ngày xuống thì thực hiện ép kiểu so sánh
        if (startDate != null && !startDate.isEmpty() && endDate != null && !endDate.isEmpty()) {
            
            // Chuyển chuỗi "YYYY-MM-DD" từ giao diện thành mốc 00:00:00 đầu ngày
            LocalDateTime startDateTime = LocalDate.parse(startDate).atStartOfDay();
            
            // Chuyển chuỗi "YYYY-MM-DD" từ giao diện thành mốc 23:59:59 cuối ngày
            LocalDateTime endDateTime = LocalDate.parse(endDate).atTime(LocalTime.MAX);

            return sessions.stream()
                    .filter(s -> {
                        LocalDateTime targetTime = s.getStartTime(); // Lấy thời gian bắt đầu của phiên
                        if (targetTime == null) return false;
                        
                        // Chỉ giữ lại những phiên nằm trong khoảng thời gian được chọn
                        return !targetTime.isBefore(startDateTime) && !targetTime.isAfter(endDateTime);
                    })
                    .map(s -> this.convertToDTO(s))
                    .collect(Collectors.toList());
        }

        // 3. Fallback mặc định: Nếu không chọn mốc thời gian, hiển thị toàn bộ lịch sử
        return sessions.stream()
                       .map(s -> this.convertToDTO(s))
                       .collect(Collectors.toList());
    }

    // =========================================================================
    // GIỮ NGUYÊN HOÀN TOÀN CÁC HÀM XỬ LÝ ẢNH CƠ BẢN CỦA TOÀN PHÍA DƯỚI ĐỂ TRÁNH LỖI:
    // =========================================================================
    public List<String> getSessionImageUrls(String sessionId) throws IllegalArgumentException, Exception {
        if (sessionId == null || !sessionId.matches("\\d{8}_\\d{6}")) {
            throw new IllegalArgumentException("Định dạng ID phải là YYYYMMDD_HHmmss");
        }

        String[] parts = sessionId.split("_");
        String datePart = parts[0]; 
        String timePart = parts[1]; 

        LocalDate date = LocalDate.parse(datePart, DateTimeFormatter.ofPattern("yyyyMMdd"));
        LocalTime time = LocalTime.parse(timePart, DateTimeFormatter.ofPattern("HHmmss"));

        long sessionSeconds = LocalDateTime.of(date, time).toEpochSecond(ZoneOffset.ofHours(7));

        String year = datePart.substring(0, 4);
        String month = datePart.substring(4, 6);
        String day = datePart.substring(6, 8);

        File dateDir = Paths.get(STORAGE_PATH, year, month, day).toFile();
        if (!dateDir.exists() || !dateDir.isDirectory()) return null;

        File[] folders = dateDir.listFiles(File::isDirectory);
        if (folders == null || folders.length == 0) return null;

        File matchedFolder = Arrays.stream(folders)
                .filter(folder -> isWithin2Seconds(folder.getName(), date, sessionSeconds))
                .findFirst()
                .orElse(null);

        if (matchedFolder == null) return null;

        File rawDir = new File(matchedFolder, "raw");
        if (!rawDir.exists() || !rawDir.isDirectory()) return null;

        String[] files = rawDir.list((dir, name) ->
                name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".png")
        );

        if (files == null || files.length == 0) return Collections.emptyList();

        return Arrays.stream(files)
                .map(f -> "/sessions/display-image?fullPath=" +
                        URLEncoder.encode(new File(rawDir, f).getAbsolutePath(), StandardCharsets.UTF_8))
                .collect(Collectors.toList());
    }

    private boolean isWithin2Seconds(String folderName, LocalDate date, long sessionSeconds) {
        try {
            String[] p = folderName.split("_");
            if (p.length < 3) return false;

            int h = Integer.parseInt(p[0]);
            int m = Integer.parseInt(p[1]);
            int s = Integer.parseInt(p[2]);

            LocalTime folderTime = LocalTime.of(h, m, s);
            long folderSeconds = LocalDateTime.of(date, folderTime).toEpochSecond(ZoneOffset.ofHours(7));

            return Math.abs(folderSeconds - sessionSeconds) <= 2;
        } catch (Exception e) {
            return false;
        }
    }
    
    // Giữ nguyên hàm convertToDTO(Session s) của bạn ở đây...
}