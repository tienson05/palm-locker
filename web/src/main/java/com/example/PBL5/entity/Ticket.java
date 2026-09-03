package com.example.PBL5.entity;

import jakarta.persistence.*;

import java.time.LocalDateTime;

@Entity
@Table(name ="ticket")
public class Ticket {
    @Id
    private String id;
    @Column(name ="created_at")
    private LocalDateTime created_at;
    @Column (name ="reason")
    private String reason;
    @ManyToOne
    @JoinColumn(name ="session_id")
    private Session session;
    @ManyToOne
    @JoinColumn(name ="locker_id")
    private Locker locker;
    public Ticket() {}

    public Ticket(String id, LocalDateTime created_at, String reason, Session session) {
        this.id = id;
        this.created_at = created_at;
        this.reason = reason;
        this.session = session;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public LocalDateTime getCreated_at() {
        return created_at;
    }

    public void setCreated_at(LocalDateTime created_at) {
        this.created_at = created_at;
    }

    public Locker getLocker() {
        return locker;
    }

    public void setLocker(Locker locker) {
        this.locker = locker;
    }

    public String getReason() {
        return reason;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }

    public Session getSession() {
        return session;
    }

    public void setSession(Session session) {
        this.session = session;
    }
}
