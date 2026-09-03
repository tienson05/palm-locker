package com.example.PBL5.controller;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.example.PBL5.dto.SessionResponse;
import com.example.PBL5.service.SessionService;

@RestController
@RequestMapping("/sessions")
@CrossOrigin(origins = "*")
public class SessionController {

    private final SessionService sessionService;

    public SessionController(SessionService sessionService) {
        this.sessionService = sessionService;
    }

    @GetMapping
    public List<SessionResponse> getAllSessions() {
        return sessionService.getAllSessions();
    }

    @GetMapping("/{id}")
    public SessionResponse getSessionById(@PathVariable String id) {
        return sessionService.getSessionById(id);
    }

    @GetMapping("/{lockerId}/current-session")
    public SessionResponse getCurrentSession(@PathVariable String lockerId) {
        return sessionService.getCurrentSession(lockerId);
    }

    @GetMapping("/search")
public List<SessionResponse> searchSessions(
        @RequestParam(required = false) String lockerId,
        @RequestParam(required = false) String status,
        @RequestParam(defaultValue = "startTime") String sortBy,
        @RequestParam(required = false) String startDate,
        @RequestParam(required = false) String endDate) {
    return sessionService.searchSessions(lockerId, status, sortBy, startDate, endDate);
}

    @GetMapping("/images-by-session")
    public ResponseEntity<?> getSessionImages(@RequestParam String sessionId) {
        try {
            List<String> imageUrls = sessionService.getSessionImageUrls(sessionId);
            
            if (imageUrls == null) {
                return ResponseEntity.status(404).body("⚠️ Không tìm thấy tài nguyên thư mục ảnh tương ứng!");
            }
            
            return ResponseEntity.ok(imageUrls);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Lỗi hệ thống khi quét ảnh: " + e.getMessage());
        }
    }

    @GetMapping("/display-image")
    public ResponseEntity<Resource> displayImage(@RequestParam String fullPath) {
        try {
            Path path = Paths.get(fullPath);
            Resource resource = new UrlResource(path.toUri());
            return ResponseEntity.ok()
                    .contentType(MediaType.IMAGE_JPEG)
                    .body(resource);
        } catch (Exception e) {
            return ResponseEntity.notFound().build();
        }
    }
}