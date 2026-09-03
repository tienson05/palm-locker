package com.example.PBL5.entity;

import java.time.LocalDateTime;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

@Entity
@Table(name = "session")
public class Session {
    
    @Id
    private String id;

    @Column(name = "palm_hash")
    private String palmHash;

    @Column(name = "start_time")
    private LocalDateTime startTime; 

    @Column(name = "end_time")
    private LocalDateTime endTime;

    @Column(name = "status")
    private String status;

    @ManyToOne
    @JoinColumn(name = "locker_id")
    private Locker locker;

    // Constructor không tham số bắt buộc của JPA
    public Session() {
    }

    // Constructor có tham số đã chuẩn hóa tên biến camelCase
    public Session(String id, String palmHash, LocalDateTime startTime, LocalDateTime endTime, String status, Locker locker) {
        this.id = id;
        this.palmHash = palmHash;
        this.startTime = startTime;
        this.endTime = endTime;
        this.status = status;
        this.locker = locker;
    }

    // ==========================================
    // HỆ THỐNG GETTER & SETTER CHUẨN CAMELCASE
    // ==========================================
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getPalmHash() { // Đổi từ getPalm_hash
        return palmHash;
    }

    public void setPalmHash(String palmHash) { // Đổi từ setPalm_hash
        this.palmHash = palmHash;
    }

    public LocalDateTime getStartTime() {
        return startTime;
    }

    public void setStartTime(LocalDateTime startTime) {
        this.startTime = startTime;
    }

    public LocalDateTime getEndTime() {
        return endTime;
    }

    public void setEndTime(LocalDateTime endTime) {
        this.endTime = endTime;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public Locker getLocker() {
        return locker;
    }

    public void setLocker(Locker locker) {
        this.locker = locker;
    }


}