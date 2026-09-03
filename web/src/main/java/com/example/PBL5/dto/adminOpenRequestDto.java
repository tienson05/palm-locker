package com.example.PBL5.dto;

public class adminOpenRequestDto {
    private String lockerId;
    private String reason;

    public adminOpenRequestDto() {
    }

    public adminOpenRequestDto(String lockerId, String reason) {
        this.lockerId = lockerId;
        this.reason = reason;
    }

    public String getLockerId() {
        return lockerId;
    }

    public void setLockerId(String lockerId) {
        this.lockerId = lockerId;
    }

    public String getReason() {
        return reason;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }
}
