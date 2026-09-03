package com.example.PBL5.controller;

import java.util.List;
import java.util.Map;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.example.PBL5.dto.adminOpenRequestDto;
import com.example.PBL5.dto.createLocker;
import com.example.PBL5.dto.updateLocker;
import com.example.PBL5.entity.Locker;
import com.example.PBL5.service.LockerService;

@RestController
@RequestMapping("/lockers")
@CrossOrigin(origins = "*") // Đảm bảo cấu hình CORS để Frontend fetch không bị chặn
public class LockerController {

    private final LockerService lockerService;

    // Tiêm các dependency cần thiết thông qua Constructor
    public LockerController(LockerService lockerService) {
        this.lockerService = lockerService;
    }

    @GetMapping
    public List<Locker> getAllLockers() {
        return lockerService.getAllLockers();
    }

    @PostMapping
    public Locker createLocker(@RequestBody createLocker request) {
        Locker locker = new Locker();
        locker.setLocation(request.getLocation());
        locker.setStatus("available");
        return lockerService.createLocker(locker);
    }

    @GetMapping("/{id}")
    public Locker getLockerById(@PathVariable String id) {
        return lockerService.getLockerById(id);
    }

    @PutMapping("/{id}")
    public Locker updateLocker(@PathVariable String id, @RequestBody updateLocker request) {
        return lockerService.updateLocker(id, request);
    }

    @DeleteMapping("/{id}")
    public String deleteLocker(@PathVariable String id) {
        lockerService.deleteLocker(id);
        return "deleted";
    }

    @GetMapping("/search")
    public List<Locker> searchLockers(
            @RequestParam(defaultValue = "") String keyword,
            @RequestParam(defaultValue = "ALL") String status) {
        return lockerService.searchLockers(keyword, status);
    }

    @PostMapping("/open")
    public String openLocker(@RequestBody adminOpenRequestDto request) {
        return lockerService.adminOpenLocker(request);
    }

   @GetMapping("/dashboard/stats")
    public ResponseEntity<Map<String, Object>> getDashboardStats() {
        return ResponseEntity.ok(lockerService.getDashboardStatistics());
    }
    
}