package com.example.PBL5.utils;

public class IdGenerator {
    public static synchronized String generateId(String lastId, String prefix) {
        if (lastId == null || lastId.isEmpty()) {
            return prefix + "0000001";
        }

        String numberPart = lastId.substring(prefix.length());
        int number = Integer.parseInt(numberPart);
        number++;

        String newNumber = String.format("%07d", number);
        return prefix + newNumber;
    }
}
