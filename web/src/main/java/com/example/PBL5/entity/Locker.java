package com.example.PBL5.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;
import jakarta.persistence.Id;

@Entity
@Table(name =  "locker")

public class Locker {
    @Id
    private String id;
    @Column(name = "status")
    private String status;
    @Column(name = "location")
    private String location;
    public Locker() {

    }
    public Locker(String id, String status, String location) {
        this.id = id;
        this.status = status;
        this.location = location;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getLocation() {
        return location;
    }

    public void setLocation(String location) {
        this.location = location;
    }
}
